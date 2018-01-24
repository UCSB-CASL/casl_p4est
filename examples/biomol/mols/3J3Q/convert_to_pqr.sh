#!/bin/sh
for filename in ./*pdb; do
	../pdb2pqr/pdb2pqr --ff=parse "$filename" "./pqr/$(basename "$filename" .pdb).pqr"
done
