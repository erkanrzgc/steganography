#!/bin/sh
# Keep the JVM within ToolRunner's 768 MiB address-space budget.
# Bound native allocator arenas and CPU-derived JVM thread pools as well as heap.
export MALLOC_ARENA_MAX=2
exec java -Djava.awt.headless=true -Duser.home=/tmp -Xmx128m -XX:+UseSerialGC \
    -XX:ActiveProcessorCount=2 \
    -XX:ReservedCodeCacheSize=64m -XX:CompressedClassSpaceSize=64m \
    -jar /usr/share/openstego/lib/openstego.jar "$@"
