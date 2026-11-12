package dev.bharath.gateway;

import static org.assertj.core.api.Assertions.assertThat;

import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.http.HttpStatus;
import org.springframework.test.annotation.DirtiesContext;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@DirtiesContext(classMode = DirtiesContext.ClassMode.AFTER_EACH_TEST_METHOD)
class GatewayTest {

    enum Mode { OK, ERROR, SLOW, NO_HISTORY }

    static final AtomicReference<Mode> mode = new AtomicReference<>(Mode.OK);
    static final AtomicInteger modelCalls = new AtomicInteger();
    static final HttpServer fake;
    static final Path fallbackFile;

    static {
        try {
            fallbackFile = Files.createTempFile("fallback", ".csv");
            Files.writeString(fallbackFile, "zone,trips\n132,41\n161,17\n");
            fake = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
            fake.createContext("/v1/forecast", ex -> {
                modelCalls.incrementAndGet();
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
            fake.start();
        } catch (IOException e) {
            throw new ExceptionInInitializerError(e);
        }
    }

    @DynamicPropertySource
    static void props(DynamicPropertyRegistry r) {
        r.add("gateway.model-url", () -> "http://127.0.0.1:" + fake.getAddress().getPort());
        r.add("gateway.fallback-file", () -> fallbackFile.toString());
    }

    @Autowired
    TestRestTemplate http;

    @BeforeEach
    void reset() {
        mode.set(Mode.OK);
        modelCalls.set(0);
    }

    @Test
    void servesTheModelWhenItIsHealthy() {
        var res = http.getForEntity("/v1/forecast/132", ForecastService.Result.class);
        assertThat(res.getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat(res.getBody().source()).isEqualTo("model");
        assertThat(res.getBody().trips()).isEqualTo(55.5);
    }

    @Test
    void fallsBackToSeasonalNaiveOnModelError() {
        mode.set(Mode.ERROR);
        var body = http.getForEntity("/v1/forecast/132", ForecastService.Result.class).getBody();
        assertThat(body.source()).isEqualTo("seasonal_naive_fallback");
        assertThat(body.trips()).isEqualTo(41.0);
    }

    @Test
    void fallsBackWhenTheModelIsSlowerThanTheTimeout() {
        mode.set(Mode.SLOW);
        long t0 = System.nanoTime();
        var body = http.getForEntity("/v1/forecast/161", ForecastService.Result.class).getBody();
        long ms = (System.nanoTime() - t0) / 1_000_000;
        assertThat(body.source()).isEqualTo("seasonal_naive_fallback");
        // the stub takes 2 s; answering well before that means the gateway cut the call off at its 150 ms timeout
        assertThat(ms).isLessThan(1500);
    }

    @Test
    void breakerOpensAndStopsCallingTheModel() {
        mode.set(Mode.ERROR);
        for (int i = 0; i < 20; i++) {
            http.getForEntity("/v1/forecast/132", String.class);
        }
        assertThat(modelCalls.get()).isEqualTo(20);
        mode.set(Mode.OK);
        var body = http.getForEntity("/v1/forecast/132", ForecastService.Result.class).getBody();
        assertThat(body.source()).isEqualTo("seasonal_naive_fallback");
        assertThat(modelCalls.get()).isEqualTo(20);
    }

    @Test
    void noHistoryIsA404AndDoesNotTripTheBreaker() {
        mode.set(Mode.NO_HISTORY);
        for (int i = 0; i < 30; i++) {
            assertThat(http.getForEntity("/v1/forecast/132", String.class).getStatusCode()).isEqualTo(HttpStatus.NOT_FOUND);
        }
        assertThat(modelCalls.get()).isEqualTo(30);
    }

    @Test
    void rejectsZonesOutsideTheLookupRange() {
        assertThat(http.getForEntity("/v1/forecast/0", String.class).getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
        assertThat(http.getForEntity("/v1/forecast/266", String.class).getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
    }
}
