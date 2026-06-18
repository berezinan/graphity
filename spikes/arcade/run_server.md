# Running ArcadeDB for the spike

ArcadeDB 26.6.1, JDK 25, Windows. The bundled `bin/server.sh` builds a
`:`-separated classpath that fails on Windows (`ClassNotFoundException:
com.arcadedb.server.ArcadeDBServer`). Launch the jar directly instead — Java
expands the `lib/*` wildcard itself, so no path separator is involved.

## Download

```bash
curl -sL -o arcadedb.tar.gz \
  https://github.com/ArcadeData/arcadedb/releases/download/26.6.1/arcadedb-26.6.1-minimal.tar.gz
tar -xzf arcadedb.tar.gz   # -> arcadedb-26.6.1/
```

The `-minimal` package already contains `lib/arcadedb-server-26.6.1.jar`.

## Start (from inside arcadedb-26.6.1/)

```bash
java -Xms512M -Xmx2G \
  --add-opens java.base/java.util.concurrent.atomic=ALL-UNNAMED \
  --add-opens java.base/java.nio.channels.spi=ALL-UNNAMED \
  --add-opens java.base/java.lang=ALL-UNNAMED \
  --add-modules jdk.incubator.vector \
  --enable-native-access=ALL-UNNAMED \
  -Djava.awt.headless=true -Dfile.encoding=UTF8 \
  -Djava.util.logging.config.file=config/arcadedb-log.properties \
  -Darcadedb.server.rootPassword=playwithdata \
  -cp "lib/*" \
  com.arcadedb.server.ArcadeDBServer > server.log 2>&1 &
```

HTTP API on `http://127.0.0.1:2480` (Studio UI at the same address). Readiness:

```bash
curl -s -o /dev/null -w "%{http_code}" -u root:playwithdata http://127.0.0.1:2480/api/v1/ready
# 204 == ready
```

Credentials used by `arcade_backend.py`: `root` / `playwithdata`.

## Stop

```bash
# find and kill the java process
ps -ef | grep ArcadeDBServer | grep -v grep
```
