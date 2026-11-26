package com.stylelens.catalog.config;

import com.stylelens.catalog.inference.InferenceClient;
import java.net.http.HttpClient;
import java.time.Duration;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.CacheControl;
import org.springframework.http.client.JdkClientHttpRequestFactory;
import org.springframework.web.client.RestClient;
import org.springframework.web.servlet.config.annotation.ResourceHandlerRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration(proxyBeanMethods = false)
@EnableConfigurationProperties(StylelensProperties.class)
public class WebConfig implements WebMvcConfigurer {

    private final StylelensProperties properties;

    public WebConfig(StylelensProperties properties) {
        this.properties = properties;
    }

    /** Photos du catalogue ({@code images/<id>.jpg}), servies telles quelles et mises en cache. */
    @Override
    public void addResourceHandlers(ResourceHandlerRegistry registry) {
        String location = properties.images().dir().resolve("images").toAbsolutePath().normalize().toUri().toString();
        registry.addResourceHandler("/images/**")
                .addResourceLocations(location.endsWith("/") ? location : location + "/")
                .setCacheControl(CacheControl.maxAge(Duration.ofDays(7)).cachePublic());
    }

    @Bean
    InferenceClient inferenceClient() {
        StylelensProperties.Inference config = properties.inference();
        HttpClient http = HttpClient.newBuilder()
                .version(HttpClient.Version.HTTP_1_1) // uvicorn : pas de HTTP/2 en clair
                .connectTimeout(config.connectTimeout())
                .build();
        JdkClientHttpRequestFactory factory = new JdkClientHttpRequestFactory(http);
        factory.setReadTimeout(config.readTimeout());
        return new InferenceClient(RestClient.builder().baseUrl(config.baseUrl()).requestFactory(factory).build());
    }
}
