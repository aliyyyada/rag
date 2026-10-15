#!/usr/bin/env bash
# Готовит всё для запуска на ПК без интернета. Запускать на машине С интернетом и Docker (Mac или Linux):
#     bash scripts/make_offline_bundle.sh
# Результат: папка models/ (модели) и файлы offline/rag-app.tar, offline/ollama.tar (образы Docker).
# После этого всю папку проекта копируют на заводской ПК (флешка, сетевой диск).
#
# Заводские ПК на Intel, поэтому образы собираются под linux/amd64 — даже на Mac с M1/M2/M3.
set -euo pipefail
cd "$(dirname "$0")/.."

PLATFORM="${PLATFORM:-linux/amd64}"
OLLAMA_IMAGE="ollama/ollama:latest"
APP_IMAGE="rag-docs:latest"

[ -f .env ] || cp .env.example .env
env_value() { grep -E "^$1=" .env | tail -1 | cut -d= -f2- || true; }
LLM_MODEL="$(env_value LLM_MODEL)"; LLM_MODEL="${LLM_MODEL:-qwen3:4b}"
EMBED_MODEL="$(env_value EMBED_MODEL)"; EMBED_MODEL="${EMBED_MODEL:-bge-m3}"
LLM_MODEL_LITE="$(env_value LLM_MODEL_LITE)"; LLM_MODEL_LITE="${LLM_MODEL_LITE:-qwen2.5:1.5b}"

echo "==> 1/4 Скачиваю модели $LLM_MODEL, $LLM_MODEL_LITE (запасная) и $EMBED_MODEL в папку models/ (около 4.7 ГБ)"
mkdir -p models
docker compose up -d ollama
docker compose run --rm ollama-pull
docker compose stop ollama

# Образы собираются buildx сразу в tar-файлы под нужную платформу. Через локальное хранилище Docker
# не идём: на Mac оно хранит образ для всех платформ сразу и показывает архитектуру самого Mac.
mkdir -p offline
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

# build_tar <тег> <файл.tar> <контекст>: собирает образ под $PLATFORM прямо в файл.
# Если сборщик Docker по умолчанию не умеет сохранять в файл (старое хранилище образов),
# создаётся отдельный сборщик buildx на основе контейнера.
# (без массивов: на Mac стоит bash 3.2, где пустой массив при set -u считается ошибкой)
BUILDER=""
build_tar() {
  if ! docker buildx build ${BUILDER:+--builder "$BUILDER"} --platform "$PLATFORM" -t "$1" \
       --output "type=docker,dest=$2" "$3"; then
    [ -z "$BUILDER" ] || return 1
    echo "    Сборщик по умолчанию не справился, создаю отдельный (rag-builder)…"
    docker buildx inspect rag-builder >/dev/null 2>&1 || docker buildx create --name rag-builder --driver docker-container >/dev/null
    BUILDER="rag-builder"
    docker buildx build --builder "$BUILDER" --platform "$PLATFORM" -t "$1" --output "type=docker,dest=$2" "$3"
  fi
}

echo "==> 2/4 Собираю образ приложения под $PLATFORM"
build_tar "$APP_IMAGE" offline/rag-app.tar .

echo "==> 3/4 Готовлю образ Ollama под $PLATFORM"
echo "FROM $OLLAMA_IMAGE" > "$TMP_DIR/Dockerfile"
build_tar "$OLLAMA_IMAGE" offline/ollama.tar "$TMP_DIR"

echo "==> 4/4 Проверяю архитектуру образов"
want_arch="${PLATFORM#*/}"
for f in offline/rag-app.tar offline/ollama.tar; do
  cfg="$(tar -xOf "$f" manifest.json | grep -o '"Config":"[^"]*"' | head -1 | cut -d'"' -f4)"
  arch="$(tar -xOf "$f" "$cfg" | grep -o '"architecture":"[^"]*"' | head -1 | cut -d'"' -f4)"
  if [ "$arch" != "$want_arch" ]; then
    echo "ОШИБКА: $f собран под '$arch', а нужен '$want_arch'" >&2
    exit 1
  fi
  echo "    $f: $arch — ок"
done
rm -f offline/rag-images.tar  # файл от прошлой версии скрипта

echo
echo "Готово. Скопируйте ВСЮ папку проекта на заводской ПК (models/ и offline/ обязательно) и там выполните:"
echo "    docker load -i offline/rag-app.tar"
echo "    docker load -i offline/ollama.tar"
echo "    docker compose up -d --no-build"
du -sh models offline/*.tar 2>/dev/null || true
