package dev.bharath.gateway;

import static org.assertj.core.api.Assertions.assertThat;

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

    static final FakeModel model = new FakeModel();

    @DynamicPropertySource
    static void props(DynamicPropertyRegistry r) {
        r.add("gateway.model-url", model::url);
        r.add("gateway.fallback-file", () -> model.fallbackFile.toString());
        // The production timeout is 150 ms. A cold JVM on a small host can miss that on the very first call,
        // which says nothing about the gateway logic these tests check, so they run with a generous one.
        // GatewaySlowModelTest covers the timeout itself.
        r.add("gateway.model-timeout", () -> "5s");
    }

    @Autowired
    TestRestTemplate http;

    @BeforeEach
    void reset() {
        model.reset();
    }

    @Test
    void servesTheModelWhenItIsHealthy() {
        http.getForEntity("/v1/forecast/161", String.class); // warm the HTTP client and the JSON path
        var res = http.getForEntity("/v1/forecast/132", ForecastService.Result.class);
        assertThat(res.getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat(res.getBody().source()).isEqualTo("model");
        assertThat(res.getBody().trips()).isEqualTo(55.5);
    }

    @Test
    void fallsBackToSeasonalNaiveOnModelError() {
        model.mode.set(FakeModel.Mode.ERROR);
        var body = http.getForEntity("/v1/forecast/132", ForecastService.Result.class).getBody();
        assertThat(body.source()).isEqualTo("seasonal_naive_fallback");
        assertThat(body.trips()).isEqualTo(41.0);
    }

    @Test
    void breakerOpensAndStopsCallingTheModel() {
        model.mode.set(FakeModel.Mode.ERROR);
        for (int i = 0; i < 20; i++) {
            http.getForEntity("/v1/forecast/132", String.class);
        }
        assertThat(model.calls.get()).isEqualTo(20);
        model.mode.set(FakeModel.Mode.OK);
        var body = http.getForEntity("/v1/forecast/132", ForecastService.Result.class).getBody();
        assertThat(body.source()).isEqualTo("seasonal_naive_fallback");
        assertThat(model.calls.get()).isEqualTo(20);
    }

    @Test
    void noHistoryIsA404AndDoesNotTripTheBreaker() {
        model.mode.set(FakeModel.Mode.NO_HISTORY);
        for (int i = 0; i < 30; i++) {
            assertThat(http.getForEntity("/v1/forecast/132", String.class).getStatusCode()).isEqualTo(HttpStatus.NOT_FOUND);
        }
        assertThat(model.calls.get()).isEqualTo(30);
    }

    @Test
    void rejectsZonesOutsideTheLookupRange() {
        assertThat(http.getForEntity("/v1/forecast/0", String.class).getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
        assertThat(http.getForEntity("/v1/forecast/266", String.class).getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
    }
}
