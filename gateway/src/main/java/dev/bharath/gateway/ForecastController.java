package dev.bharath.gateway;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class ForecastController {

    private final ForecastService service;

    public ForecastController(ForecastService service) {
        this.service = service;
    }

    @GetMapping("/v1/forecast/{zone}")
    public ResponseEntity<?> forecast(@PathVariable int zone) {
        if (zone < 1 || zone > 265) {
            return ResponseEntity.badRequest().body("zone must be between 1 and 265");
        }
        return service.forecast(zone)
                .<ResponseEntity<?>>map(ResponseEntity::ok)
                .orElseGet(() -> ResponseEntity.status(HttpStatus.NOT_FOUND).body("no forecast for zone " + zone));
    }
}
