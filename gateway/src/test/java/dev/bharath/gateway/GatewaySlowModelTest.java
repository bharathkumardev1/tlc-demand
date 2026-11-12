package dev.bharath.gateway;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;

/** Runs with the production 150 ms model timeout, against a model stub that takes 2 s to answer. */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
class GatewaySlowModelTest {

    static final FakeModel model = new FakeModel();

    @DynamicPropertySource
    static void props(DynamicPropertyRegistry r) {
        r.add("gateway.model-url", model::url);
        r.add("gateway.fallback-file", () -> model.fallbackFile.toString());
        r.add("gateway.model-timeout", () -> "150ms");
    }

    @Autowired
    TestRestTemplate http;

    @Test
    void fallsBackWhenTheModelIsSlowerThanTheTimeout() {
        model.mode.set(FakeModel.Mode.SLOW);
        long t0 = System.nanoTime();
        var body = http.getForEntity("/v1/forecast/161", ForecastService.Result.class).getBody();
        long ms = (System.nanoTime() - t0) / 1_000_000;
        assertThat(body.source()).isEqualTo("seasonal_naive_fallback");
        assertThat(body.trips()).isEqualTo(17.0);
        assertThat(model.calls.get()).isEqualTo(1);
        // The stub needs 2 s. Answering well before that means the gateway cut the call off at its timeout.
        assertThat(ms).isLessThan(1500);
    }
}
