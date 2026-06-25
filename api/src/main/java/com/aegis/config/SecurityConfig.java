package com.aegis.config;

import java.util.Collection;
import java.util.List;
import java.util.stream.Stream;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.boot.autoconfigure.condition.ConditionalOnExpression;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.convert.converter.Converter;
import org.springframework.http.HttpMethod;
import org.springframework.security.authentication.AbstractAuthenticationToken;
import org.springframework.security.config.annotation.method.configuration.EnableMethodSecurity;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.oauth2.core.DelegatingOAuth2TokenValidator;
import org.springframework.security.oauth2.core.OAuth2TokenValidator;
import org.springframework.security.oauth2.jwt.Jwt;
import org.springframework.security.oauth2.jwt.JwtClaimNames;
import org.springframework.security.oauth2.jwt.JwtDecoder;
import org.springframework.security.oauth2.jwt.JwtDecoders;
import org.springframework.security.oauth2.jwt.JwtValidators;
import org.springframework.security.oauth2.jwt.NimbusJwtDecoder;
import org.springframework.security.oauth2.server.resource.authentication.JwtAuthenticationConverter;
import org.springframework.security.oauth2.server.resource.authentication.JwtGrantedAuthoritiesConverter;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.web.cors.CorsConfiguration;
import org.springframework.web.cors.CorsConfigurationSource;
import org.springframework.web.cors.UrlBasedCorsConfigurationSource;

/** Security model (spec: trustworthy AI in a regulated domain).
 *
 *  The public read-only demo (flags, explain, summary, investigate, metrics, adversarial) is open so
 *  recruiters can click it with zero friction. The analyst workflow ({@code /api/cases/**},
 *  {@code /api/me}) requires a valid Entra ID (Azure AD) access token with an app role.
 *
 *  Activation is env-driven and FAILSAFE: a JWT resource server is wired only when AEGIS_OIDC_ISSUER
 *  is set. With it unset (e.g. the current deployment) there is no decoder and every request is
 *  permitted — so adding security can never lock the demo out by accident. */
@Configuration
@EnableWebSecurity
@EnableMethodSecurity     // enables @PreAuthorize on the (forthcoming) case-management endpoints
public class SecurityConfig {

    /** Read-only endpoints that stay public even when auth is enabled. */
    private static final String[] PUBLIC_GET = {
            "/api/datasets", "/api/flags/**", "/api/explain/**", "/api/summary/**",
            "/api/investigate/**", "/api/graph/**", "/api/metrics/**"
    };

    /** Built only when an issuer is configured; presence of this bean == "auth enabled". */
    @Bean
    @ConditionalOnExpression("'${AEGIS_OIDC_ISSUER:}' != ''")
    JwtDecoder jwtDecoder(@org.springframework.beans.factory.annotation.Value("${AEGIS_OIDC_ISSUER}") String issuer,
                          @org.springframework.beans.factory.annotation.Value("${AEGIS_OIDC_AUDIENCE:}") String audience) {
        NimbusJwtDecoder decoder = (NimbusJwtDecoder) JwtDecoders.fromIssuerLocation(issuer);
        OAuth2TokenValidator<Jwt> base = JwtValidators.createDefaultWithIssuer(issuer);
        OAuth2TokenValidator<Jwt> validator = audience.isBlank() ? base
                : new DelegatingOAuth2TokenValidator<>(base, audienceValidator(audience));
        decoder.setJwtValidator(validator);
        return decoder;
    }

    @Bean
    SecurityFilterChain filterChain(HttpSecurity http, ObjectProvider<JwtDecoder> jwtDecoder) throws Exception {
        JwtDecoder decoder = jwtDecoder.getIfAvailable();
        boolean authEnabled = decoder != null;

        http
            .csrf(csrf -> csrf.disable())                       // stateless token API, no cookies
            .cors(cors -> cors.configurationSource(corsSource()))
            .sessionManagement(s -> s.sessionCreationPolicy(SessionCreationPolicy.STATELESS))
            .authorizeHttpRequests(reg -> {
                reg.requestMatchers(HttpMethod.OPTIONS, "/**").permitAll();          // CORS preflight
                reg.requestMatchers("/actuator/health/**", "/actuator/info").permitAll();
                reg.requestMatchers(HttpMethod.GET, PUBLIC_GET).permitAll();         // read-only demo
                reg.requestMatchers(HttpMethod.POST, "/api/adversarial/run").permitAll();
                if (authEnabled) {
                    reg.requestMatchers("/api/cases/**").hasAnyRole("ANALYST", "REVIEWER", "ADMIN");
                    reg.anyRequest().authenticated();           // /api/me, writes, anything else
                } else {
                    reg.anyRequest().permitAll();               // no auth configured -> fully open demo
                }
            });

        if (authEnabled) {
            http.oauth2ResourceServer(oauth -> oauth.jwt(jwt ->
                    jwt.decoder(decoder).jwtAuthenticationConverter(authoritiesConverter())));
        }
        return http.build();
    }

    /** Maps Entra ID app roles (the {@code roles} claim) and scopes ({@code scp}) to authorities;
     *  app roles become {@code ROLE_*} so hasRole(...) / @PreAuthorize("hasRole(...)") work. */
    private Converter<Jwt, AbstractAuthenticationToken> authoritiesConverter() {
        JwtGrantedAuthoritiesConverter scopes = new JwtGrantedAuthoritiesConverter();   // SCOPE_*
        JwtAuthenticationConverter converter = new JwtAuthenticationConverter();
        converter.setJwtGrantedAuthoritiesConverter(jwt -> {
            List<String> roles = jwt.getClaimAsStringList("roles");
            Stream<GrantedAuthority> roleAuth = roles == null ? Stream.empty()
                    : roles.stream().map(r -> new SimpleGrantedAuthority("ROLE_" + r));
            Collection<GrantedAuthority> all = scopes.convert(jwt);
            return Stream.concat(all.stream(), roleAuth).distinct().toList();
        });
        return converter;
    }

    private OAuth2TokenValidator<Jwt> audienceValidator(String audience) {
        return jwt -> {
            List<String> aud = jwt.getClaimAsStringList(JwtClaimNames.AUD);
            return (aud != null && aud.contains(audience))
                    ? org.springframework.security.oauth2.core.OAuth2TokenValidatorResult.success()
                    : org.springframework.security.oauth2.core.OAuth2TokenValidatorResult.failure(
                        new org.springframework.security.oauth2.core.OAuth2Error("invalid_token",
                                "Required audience '" + audience + "' missing", null));
        };
    }

    /** Permissive CORS for the read-only public demo (no credentials → any origin is safe). Restrict
     *  with AEGIS_CORS_ORIGINS if ever needed. */
    @Bean
    CorsConfigurationSource corsSource() {
        CorsConfiguration cfg = new CorsConfiguration();
        cfg.setAllowedOriginPatterns(List.of("*"));
        cfg.setAllowedMethods(List.of("GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"));
        cfg.setAllowedHeaders(List.of("*"));
        cfg.setExposedHeaders(List.of("Retry-After", "X-RateLimit-Remaining"));
        UrlBasedCorsConfigurationSource src = new UrlBasedCorsConfigurationSource();
        src.registerCorsConfiguration("/**", cfg);
        return src;
    }
}
