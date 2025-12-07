# Maintainer: Your Name <your@email.com>
pkgname=nerd-dictation-gigaam
pkgver=1.0.0
pkgrel=1
pkgdesc="GigaAM ASR backend for nerd-dictation (Russian speech recognition)"
arch=('x86_64')
url="https://github.com/nerd-voice-interface/nerd-dictation"
license=('MIT')
depends=('onnxruntime' 'gcc-libs' 'glibc')
makedepends=('git' 'cmake')
provides=('nerd-dictation-backend')
source=()
sha256sums=()

# Build from local source
_srcdir="${startdir}"

prepare() {
    cd "$_srcdir"

    # Initialize submodules if needed
    if [ ! -f "ThirdParty/sherpa-onnx/CMakeLists.txt" ]; then
        git submodule update --init --recursive
    fi

    # Download model if not present
    if [ ! -f "models/sherpa-onnx-nemo-ctc-giga-am-v2-russian-2025-04-19/model.int8.onnx" ]; then
        echo "Downloading GigaAM model..."
        cd models
        curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-ctc-giga-am-v2-russian-2025-04-19.tar.bz2
        tar xjf sherpa-onnx-nemo-ctc-giga-am-v2-russian-2025-04-19.tar.bz2
        rm -f sherpa-onnx-nemo-ctc-giga-am-v2-russian-2025-04-19.tar.bz2
        cd ..
    fi
}

build() {
    cd "$_srcdir"

    cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr \
        -DBUILD_SHARED_LIBS=ON

    cmake --build build -j$(nproc)
}

package() {
    cd "$_srcdir"

    # Install shared library
    install -Dm755 build/libnerd-dictation-gigaam.so.${pkgver} \
        "${pkgdir}/usr/lib/libnerd-dictation-gigaam.so.${pkgver}"

    # Create symlinks
    ln -s libnerd-dictation-gigaam.so.${pkgver} \
        "${pkgdir}/usr/lib/libnerd-dictation-gigaam.so.1"
    ln -s libnerd-dictation-gigaam.so.${pkgver} \
        "${pkgdir}/usr/lib/libnerd-dictation-gigaam.so"

    # Install model files
    install -Dm644 models/sherpa-onnx-nemo-ctc-giga-am-v2-russian-2025-04-19/model.int8.onnx \
        "${pkgdir}/usr/share/nerd-dictation-gigaam/model/model.int8.onnx"
    install -Dm644 models/sherpa-onnx-nemo-ctc-giga-am-v2-russian-2025-04-19/tokens.txt \
        "${pkgdir}/usr/share/nerd-dictation-gigaam/model/tokens.txt"

    # Install test wav for verification
    install -Dm644 models/sherpa-onnx-nemo-ctc-giga-am-v2-russian-2025-04-19/test_wavs/example.wav \
        "${pkgdir}/usr/share/nerd-dictation-gigaam/model/test_wavs/example.wav"

    # Install C API header from submodule
    install -Dm644 ThirdParty/nerd-dictation-api/include/nerd_dictation/backend_api.h \
        "${pkgdir}/usr/include/nerd_dictation/backend_api.h"

    # Install license
    install -Dm644 ThirdParty/nerd-dictation-api/LICENSE \
        "${pkgdir}/usr/share/licenses/${pkgname}/LICENSE"
}
