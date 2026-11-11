package dev.bharath.gateway;

import io.github.resilience4j.circuitbreaker.CallNotPermittedException;
import io.github.resilience4j.circuitbreaker.CircuitBreaker;
import io.github.resilience4j.circuitbreaker.CircuitBreakerConfig;
import io.micrometer.core.instrument.MeterRegistry;
import java.util.Optional;
import org.springframework.stereotype.Service;

@Service
public class ForecastService {

    public record Result(int zone, double trips, String source, String modelVersion) {}

    private final ModelClient model;
    private final FallbackTable fallback;
    private final CircuitBreaker breaker;
    private final MeterRegistry metrics;

    public ForecastService(ModelClient model, FallbackTable fallback, GatewayProperties props, MeterRegistry metrics) {
        this.model = model;
        this.fallback = fallback;
        this.metrics = metrics;
        this.breaker = CircuitBreaker.of("model", CircuitBreakerConfig.custom()
                .slidingWindowSize(props.breakerWindow())
                .minimumNumberOfCalls(props.breakerWindow())
                .failureRateThreshold(props.breakerFailureRate())
                .waitDurationInOpenState(props.breakerOpenFor())
                .ignoreExceptions(ModelClient.NoHistory.class)
                .build());
    }

    public Optional<Result> forecast(int zone) {
        try {
            var f = breaker.executeCheckedSupplier(() -> model.forecast(zone));
            count("model");
            return Optional.of(new Result(zone, f.trips(), "model", f.modelVersion()));
        } catch (ModelClient.NoHistory e) {
            count("no_history");
            return Optional.empty();
        } catch (CallNotPermittedException e) {
            return fromFallback(zone, "breaker_open");
        } catch (Throwable e) {
            return fromFallback(zone, "model_error");
        }
    }

    private Optional<Result> fromFallback(int zone, String reason) {
        metrics.counter("gateway.fallback", "reason", reason).increment();
        var v = fallback.get(zone);
        v.ifPresentOrElse(x -> count("fallback"), () -> count("unavailable"));
        return v.map(x -> new Result(zone, x, "seasonal_naive_fallback", "none"));
    }

    private void count(String outcome) {
        metrics.counter("gateway.requests", "outcome", outcome).increment();
    }

    CircuitBreaker.State breakerState() {
        return breaker.getState();
    }

}
