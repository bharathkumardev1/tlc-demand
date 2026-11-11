package dev.bharath.gateway;

import java.time.Duration;
import java.nio.file.Path;
import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "gateway")
public record GatewayProperties(
        String modelUrl,
        Duration modelTimeout,
        Path fallbackFile,
        int breakerWindow,
        float breakerFailureRate,
        Duration breakerOpenFor) {
}
