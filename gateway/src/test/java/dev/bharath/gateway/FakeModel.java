package dev.bharath.gateway;

import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

/** A stand-in for the Python model service, shared by the gateway tests. */
final class FakeModel {

    enum Mode { OK, ERROR, SLOW, NO_HISTORY }

    final AtomicReference<Mode> mode = new AtomicReference<>(Mode.OK);
    final AtomicInteger calls = new AtomicInteger();
    final HttpServer server;
    final Path fallbackFile;

    FakeModel() {
        try {
            fallbackFile = Files.createTempFile("fallback", ".csv");
            Files.writeString(fallbackFile, "zone,trips\n132,41\n161,17\n");
            server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
            server.createContext("/v1/forecast", ex -> {
                calls.incrementAndGet();
                try {
                    if (mode.get() == Mode.SLOW) {
                        Thread.sleep(2000);
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
                int status = switch (mode.get()) {
                    case ERROR -> 500;
                    case NO_HISTORY -> 404;
                    default -> 200;
                };
                byte[] body = "{\"trips\":55.5,\"model_version\":\"v-test\",\"target\":\"2023-06-30T00:00:00\"}"
                        .getBytes(StandardCharsets.UTF_8);
                ex.sendResponseHeaders(status, status == 200 ? body.length : -1);
                if (status == 200) {
                    ex.getResponseBody().write(body);
                }
                ex.close();
            });
            server.start();
        } catch (IOException e) {
            throw new ExceptionInInitializerError(e);
        }
    }

    String url() {
        return "http://127.0.0.1:" + server.getAddress().getPort();
    }

    void reset() {
        mode.set(Mode.OK);
        calls.set(0);
    }
}
