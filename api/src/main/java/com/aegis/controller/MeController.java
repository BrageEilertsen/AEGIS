package com.aegis.controller;

import java.util.List;
import java.util.Map;
import org.springframework.security.authentication.AnonymousAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/** Identity endpoint: who is the caller and what can they do. The Angular UI calls it after an Entra
 *  login to learn the analyst's roles (and to drive which workflow actions to show). When auth is
 *  disabled (public demo) it simply reports an unauthenticated caller. */
@RestController
@RequestMapping("/api")
public class MeController {

    @GetMapping("/me")
    public Map<String, Object> me(Authentication auth) {
        boolean authenticated = auth != null && auth.isAuthenticated()
                && !(auth instanceof AnonymousAuthenticationToken);
        if (!authenticated) {
            return Map.of("authenticated", false);
        }
        List<String> authorities = auth.getAuthorities().stream()
                .map(GrantedAuthority::getAuthority).sorted().toList();
        return Map.of("authenticated", true, "name", auth.getName(), "authorities", authorities);
    }
}
