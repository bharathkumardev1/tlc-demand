package dev.bharath.gateway;

import java.io.IOException;
import java.nio.file.Files;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import org.springframework.stereotype.Component;

/** Seasonal-naive values (same slot one week earlier) per zone, written by the pipeline next to the history snapshot. */
@Component
public class FallbackTable {

    private final Map<Integer, Integer> trips = new HashMap<>();

    public FallbackTable(GatewayProperties props) throws IOException {
        var lines = Files.readAllLines(props.fallbackFile());
        for (String line : lines.subList(1, lines.size())) {
            String[] p = line.split(",");
            trips.put(Integer.parseInt(p[0].trim()), Integer.parseInt(p[1].trim()));
        }
    }

    public Optional<Integer> get(int zone) {
        return Optional.ofNullable(trips.get(zone));
    }
}
