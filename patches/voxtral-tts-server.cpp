// voxtral-tts-server.cpp
//
// A thin, zallama-shaped HTTP server on top of mudler/voxtral-tts.c's real
// inference engine (https://github.com/mudler/voxtral-tts.c, MIT).
//
// Deliberately mirrors kokoro-server's CLI and endpoint contract exactly
// (--model/--host/--port, POST /v1/audio/speech, GET /health) so it drops
// into zallama's per-model-process backend pattern with zero core changes.
//
// Unlike vbomfim/voxtral-server (which wraps this same engine but falls back
// to returning silent audio when the real engine isn't wired up), this file
// has no stub path: a model that fails to load aborts startup with a
// non-zero exit code, and a failed generation is a 500, never silence.
//
// Applied onto a clean clone of mudler/voxtral-tts.c by build-voxtral-tts.sh,
// which also fetches the two vendored single-header libraries this file
// includes (httplib.h, json.hpp) — see that script for pinned versions.

extern "C" {
#include "voxtral_tts.h"
}

#include "httplib.h"
#include "json.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace {

struct Options {
    std::string model_dir;
    std::string host = "127.0.0.1";
    int port = 9090;
};

void usage(const char *prog) {
    std::fprintf(stderr,
        "Usage: %s --model <dir> [--host <ip>] [--port <port>]\n\n"
        "  --model <dir>   Voxtral-4B-TTS model directory "
        "(consolidated.safetensors + tekken.json)\n"
        "  --host <ip>     Bind address (default: 127.0.0.1)\n"
        "  --port <port>   Bind port (default: 9090)\n",
        prog);
}

Options parse_args(int argc, char **argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&](const char *flag) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Error: %s requires a value\n", flag);
                std::exit(1);
            }
            return argv[++i];
        };
        if (arg == "--model") {
            opts.model_dir = next("--model");
        } else if (arg == "--host") {
            opts.host = next("--host");
        } else if (arg == "--port") {
            opts.port = std::atoi(next("--port").c_str());
        } else if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            std::exit(0);
        } else {
            std::fprintf(stderr, "Unknown option: %s\n", arg.c_str());
            usage(argv[0]);
            std::exit(1);
        }
    }
    if (opts.model_dir.empty()) {
        std::fprintf(stderr, "Error: --model is required\n\n");
        usage(argv[0]);
        std::exit(1);
    }
    return opts;
}

// Little-endian RIFF/WAV header for 16-bit PCM mono audio, matching the shape
// TTS_SAMPLE_RATE (24 kHz) samples from tts_generate() are already in.
std::vector<uint8_t> build_wav(const float *samples, int n_samples, int sample_rate) {
    const uint32_t data_size = static_cast<uint32_t>(n_samples) * 2;
    const uint32_t file_size = 36 + data_size;

    std::vector<uint8_t> wav(44 + data_size);
    uint8_t *p = wav.data();

    auto put_u32 = [](uint8_t *dst, uint32_t v) {
        dst[0] = v & 0xFF; dst[1] = (v >> 8) & 0xFF;
        dst[2] = (v >> 16) & 0xFF; dst[3] = (v >> 24) & 0xFF;
    };
    auto put_u16 = [](uint8_t *dst, uint16_t v) {
        dst[0] = v & 0xFF; dst[1] = (v >> 8) & 0xFF;
    };

    std::memcpy(p, "RIFF", 4);
    put_u32(p + 4, file_size);
    std::memcpy(p + 8, "WAVE", 4);
    std::memcpy(p + 12, "fmt ", 4);
    put_u32(p + 16, 16);
    put_u16(p + 20, 1);              // PCM
    put_u16(p + 22, 1);              // mono
    put_u32(p + 24, static_cast<uint32_t>(sample_rate));
    put_u32(p + 28, static_cast<uint32_t>(sample_rate * 2));
    put_u16(p + 32, 2);              // block align
    put_u16(p + 34, 16);             // bits per sample
    std::memcpy(p + 36, "data", 4);
    put_u32(p + 40, data_size);

    uint8_t *pcm = p + 44;
    for (int i = 0; i < n_samples; ++i) {
        float clamped = samples[i] < -1.0f ? -1.0f : (samples[i] > 1.0f ? 1.0f : samples[i]);
        int16_t s = static_cast<int16_t>(clamped * 32767.0f);
        pcm[i * 2] = static_cast<uint8_t>(s & 0xFF);
        pcm[i * 2 + 1] = static_cast<uint8_t>((s >> 8) & 0xFF);
    }
    return wav;
}

}  // namespace

int main(int argc, char **argv) {
    Options opts = parse_args(argc, argv);

    std::fprintf(stderr, "voxtral-tts-server: loading model from %s...\n",
                 opts.model_dir.c_str());
    tts_ctx_t *ctx = tts_load(opts.model_dir.c_str());
    if (ctx == nullptr) {
        // No stub fallback: a self-hosted deploy should fail loudly, not
        // serve silent audio and look like it's working.
        std::fprintf(stderr,
            "voxtral-tts-server: FATAL — tts_load() failed for '%s'\n",
            opts.model_dir.c_str());
        return 1;
    }
    std::fprintf(stderr, "voxtral-tts-server: model loaded.\n");

    // Engine state is not thread-safe (persistent KV-cache/scratch buffers in
    // tts_ctx_t); serialize requests, same one-request-at-a-time model
    // zallama already assumes for parakeet-server/kokoro-server.
    std::mutex gen_mutex;

    httplib::Server svr;

    svr.Get("/health", [](const httplib::Request &, httplib::Response &res) {
        res.set_content(R"({"status":"ok"})", "application/json");
    });

    svr.Post("/v1/audio/speech", [&](const httplib::Request &req, httplib::Response &res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (const json::exception &e) {
            res.status = 400;
            res.set_content(json{{"error", std::string("invalid JSON: ") + e.what()}}.dump(),
                             "application/json");
            return;
        }

        std::string text = body.value("input", "");
        std::string voice = body.value("voice", "neutral_female");
        if (text.empty()) {
            res.status = 400;
            res.set_content(json{{"error", "'input' is required"}}.dump(), "application/json");
            return;
        }

        float *samples = nullptr;
        int n_samples = 0;
        int rc;
        {
            std::lock_guard<std::mutex> lock(gen_mutex);
            rc = tts_generate(ctx, text.c_str(), voice.c_str(), &samples, &n_samples);
        }

        if (rc != 0 || samples == nullptr || n_samples <= 0) {
            res.status = 500;
            res.set_content(json{{"error", "speech generation failed"}}.dump(),
                             "application/json");
            if (samples != nullptr) std::free(samples);
            return;
        }

        std::vector<uint8_t> wav = build_wav(samples, n_samples, TTS_SAMPLE_RATE);
        std::free(samples);

        res.set_content(reinterpret_cast<const char *>(wav.data()), wav.size(), "audio/wav");
    });

    std::fprintf(stderr, "voxtral-tts-server: listening on %s:%d\n",
                 opts.host.c_str(), opts.port);
    bool ok = svr.listen(opts.host.c_str(), opts.port);

    tts_free(ctx);
    return ok ? 0 : 1;
}
