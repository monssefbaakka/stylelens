package com.stylelens.catalog.inference;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.content;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.method;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.requestTo;
import static org.springframework.test.web.client.response.MockRestResponseCreators.withServerError;
import static org.springframework.test.web.client.response.MockRestResponseCreators.withStatus;
import static org.springframework.test.web.client.response.MockRestResponseCreators.withSuccess;

import java.io.IOException;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.test.web.client.MockRestServiceServer;
import org.springframework.web.client.RestClient;

class InferenceClientTest {

    private MockRestServiceServer server;
    private InferenceClient client;

    @BeforeEach
    void setUp() {
        RestClient.Builder builder = RestClient.builder().baseUrl("http://inference");
        server = MockRestServiceServer.bindTo(builder).build();
        client = new InferenceClient(builder.build());
    }

    @Test
    void sendsImageAsMultipartAndParsesEmbedding() {
        server.expect(requestTo("http://inference/embed"))
                .andExpect(method(HttpMethod.POST))
                .andExpect(content().contentTypeCompatibleWith(MediaType.MULTIPART_FORM_DATA))
                .andExpect(content().string(org.hamcrest.Matchers.containsString("filename=\"shoe.png\"")))
                .andRespond(withSuccess("""
                        {"embedding": [0.6, 0.8], "dim": 2, "model": "resnet50", "fine_tuned": true, "elapsed_ms": 12.5}
                        """, MediaType.APPLICATION_JSON));

        Embedding e = client.embed(new byte[] {1, 2, 3}, "shoe.png", "image/png");

        assertThat(e.embedding()).containsExactly(0.6f, 0.8f);
        assertThat(e.model()).isEqualTo("resnet50");
        assertThat(e.fineTuned()).isTrue();
        assertThat(e.elapsedMs()).isEqualTo(12.5);
        server.verify();
    }

    @Test
    void clientErrorsBecomeImageRejectedWithServiceMessage() {
        server.expect(requestTo("http://inference/embed"))
                .andRespond(withStatus(HttpStatus.BAD_REQUEST)
                        .contentType(MediaType.APPLICATION_JSON)
                        .body("{\"detail\": \"Format d'image non reconnu ou fichier corrompu\"}"));

        assertThatThrownBy(() -> client.embed(new byte[] {1}, "x.jpg", "image/jpeg"))
                .isInstanceOfSatisfying(ImageRejectedException.class, e -> {
                    assertThat(e.status()).isEqualTo(400);
                    assertThat(e.getMessage()).isEqualTo("Format d'image non reconnu ou fichier corrompu");
                });
    }

    @Test
    void validationErrorsWithStructuredDetailGetGenericMessage() {
        server.expect(requestTo("http://inference/embed"))
                .andRespond(withStatus(HttpStatus.UNPROCESSABLE_CONTENT)
                        .contentType(MediaType.APPLICATION_JSON)
                        .body("{\"detail\": [{\"loc\": [\"body\", \"file\"], \"msg\": \"Field required\"}]}"));

        assertThatThrownBy(() -> client.embed(new byte[] {1}, null, null))
                .isInstanceOfSatisfying(ImageRejectedException.class,
                        e -> assertThat(e.getMessage()).isEqualTo("Image refusée par le service IA"));
    }

    @Test
    void serverErrorsAndNetworkFailuresBecomeUnavailable() {
        server.expect(requestTo("http://inference/embed")).andRespond(withServerError());
        assertThatThrownBy(() -> client.embed(new byte[] {1}, "x.jpg", "image/jpeg"))
                .isInstanceOf(InferenceUnavailableException.class)
                .hasMessageContaining("500");

        server.reset();
        server.expect(requestTo("http://inference/embed")).andRespond(request -> {
            throw new IOException("Connection refused");
        });
        assertThatThrownBy(() -> client.embed(new byte[] {1}, "x.jpg", "image/jpeg"))
                .isInstanceOf(InferenceUnavailableException.class)
                .hasMessage("Service IA injoignable");
    }

    @Test
    void readsHealth() {
        server.expect(requestTo("http://inference/health")).andRespond(withSuccess("""
                {"status": "UP", "model": "resnet50", "fine_tuned": true, "dim": 512, "image_size": 224, "device": "cpu"}
                """, MediaType.APPLICATION_JSON));

        assertThat(client.health()).isEqualTo(new ModelInfo("UP", "resnet50", true, 512, 224, "cpu"));
    }
}
