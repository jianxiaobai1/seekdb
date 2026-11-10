FROM ubuntu:24.04 AS build

RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY src /src/src
RUN g++ -O3 -std=c++17 -DNDEBUG -o /minivectordb /src/src/main.cpp

FROM ubuntu:24.04

RUN mkdir -p /data
WORKDIR /app
COPY --from=build /minivectordb /app/minivectordb

VOLUME ["/data"]
ENTRYPOINT ["/app/minivectordb"]
CMD ["--data-dir", "/data"]
