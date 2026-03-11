# Build stage
FROM rust:1.82-alpine AS builder
RUN apk add --no-cache musl-dev pkgconfig openssl-dev openssl-libs-static
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src/ ./src/
RUN cargo build --release --target x86_64-unknown-linux-musl

# Runtime stage
FROM alpine:3.20
WORKDIR /app
COPY --from=builder /app/target/x86_64-unknown-linux-musl/release/claude2openai ./
ENV LOCAL_TOKEN_COUNTING=true
EXPOSE 8788
CMD ["./claude2openai"]
