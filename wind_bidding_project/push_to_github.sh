#!/bin/bash
# Einmalig ausführen um alles ins bestehende Repo zu pushen
# Vorher: git remote muss schon gesetzt sein

git add .
git commit -m "feat: complete project skeleton with all notebooks and src modules"
git push origin main
