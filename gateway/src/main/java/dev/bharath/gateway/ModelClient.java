package dev.bharath.gateway;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import org.springframework.stereotype.Component;

/** Calls the Python model service. A 404 means the model has no history for the zone; that is not an outage. */
@Component
public class ModelClient {

    public record Forecast(double trips, String modelVersion, String target) {}

    public static class NoHistory extends RuntimeException {}

    private final HttpClient http;
    private final GatewayProperties props;
    private final ObjectMapper json = new ObjectMapper();

    public ModelClient(GatewayProperties props) {
        this.props = props;
        this.http = HttpClient.newBuilder().connectTimeout(props.modelTimeout()).build();
    }

    public Forecast forecast(int zone) throws IOException, InterruptedException {
        HttpRequest req = HttpRequest.newBuilder(URI.create(props.modelUrl() + "/v1/forecast"))
                .timeout(props.modelTimeout())
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString("{\"zone\":" + zone + "}"))
                .build();
        HttpResponse<String> res = http.send(req, HttpResponse.BodyHandlers.ofString());
        if (res.statusCode() == 404) {
            throw new NoHistory();
        }
        if (res.statusCode() != 200) {
            throw new IOException("model service returned " + res.statusCode());
        }
        JsonNode n = json.readTree(res.body());
        return new Forecast(n.get("trips").asDouble(), n.get("model_version").asText(), n.get("target").asText());
    }
}
