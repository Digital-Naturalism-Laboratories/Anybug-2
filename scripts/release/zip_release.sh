#!/bin/sh
# Zips the Anybug-2 release folder into upload/ (one file per Zenodo upload).
# Images are already JPEG, so data is stored (-0) rather than recompressed.
# Only data/, metadata/ and models/ are zipped; anything else in the folder stays local.
#   sh zip_release.sh ~/Desktop/Megabug/zenodo_Anybug-2
set -e
R=${1:?release folder}
U="$R/upload"
mkdir -p "$U"
for m in Anybug-Mixed-v2-0 Anybug-MixedNano-v2-0 Anybug-MixedOpen-v2-0 Anybug-Flat-v2-0 MBD-1-2; do
  [ -f "$R/models/$m/$m.pt" ] || { echo "missing $R/models/$m/$m.pt (pull it from the cluster first)"; exit 1; }
done
grep -q "__NANO_" "$R/README.md" && { echo "README.md still has __NANO_*__ placeholders (fill in nano energy)"; exit 1; }
cd "$R/data"     && rm -f "$U/Anybug2-Open-core.zip" && zip -q -0 -r "$U/Anybug2-Open-core.zip" Anybug2-Open-core -x '*.DS_Store'
cd "$R"          && rm -f "$U/metadata.zip"          && zip -q -r "$U/metadata.zip" metadata -x '*.DS_Store'
cd "$R/models"   && for m in *; do rm -f "$U/$m.zip"; zip -q -r "$U/$m.zip" "$m" -x '*.DS_Store'; done
cp "$R/README.md" "$R/LICENSES_AND_ATTRIBUTION.md" "$R/tools/Anybug-2-tools.zip" "$U/"
cd "$U" && shasum -a 256 * > SHA256SUMS.txt && ls -la
