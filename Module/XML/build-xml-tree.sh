#! /bin/bash

# Checked version - xmlstarlet 1.6.1
echo "digraph G {"
for LINK in $(xmlstarlet el -u $1 | grep -E -o '([^/]+)/([^/]+)$' | sed 's/\//->/g' | tr ':' '_')
do
     echo ${LINK}";"
done
echo "}"

