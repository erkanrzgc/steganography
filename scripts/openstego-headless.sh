#!/bin/sh
# Keep the JVM within ToolRunner's 768 MiB address-space budget.
exec java -Djava.awt.headless=true -Duser.home=/tmp -Xmx128m -XX:+UseSerialGC \
    -XX:ReservedCodeCacheSize=64m -XX:CompressedClassSpaceSize=64m \
    -jar /usr/share/openstego/lib/openstego.jar "$@"
