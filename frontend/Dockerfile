# Build the Angular app and serve the static bundle with nginx.
FROM node:20 AS build
WORKDIR /app
COPY package.json ./
RUN npm install
COPY . .
RUN npm run build
FROM nginx:alpine
COPY --from=build /app/dist/aegis-frontend/browser /usr/share/nginx/html
# Writes env.js at container start so the SPA learns the backend URL (AEGIS_API_URL) at runtime.
COPY docker-entrypoint.sh /docker-entrypoint.d/40-aegis-env.sh
RUN chmod +x /docker-entrypoint.d/40-aegis-env.sh
EXPOSE 80
