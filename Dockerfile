# MiniVectorDB - multi-stage build, Linux x86_64 target.
# Build stage needs network (installs toolchain); runtime stage is offline-safe.
FROM ubuntu:24.04 AS build

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      g++ cmake make binutils \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY CMakeLists.txt ./
COPY src ./src

RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
 && cmake --build build -j"$(nproc)" \
 && strip build/minivectordb

FROM ubuntu:24.04

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      libstdc++6 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=build /src/build/minivectordb /app/minivectordb

RUN mkdir -p /data
VOLUME ["/data"]

ENTRYPOINT ["/app/minivectordb"]
CMD ["--data-dir", "/data"]
