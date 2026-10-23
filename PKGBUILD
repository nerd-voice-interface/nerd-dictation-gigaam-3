# Maintainer: Your Name <your@email.com>
pkgname=nerd-dictation-gigaam
pkgver=3.0.0
pkgrel=5
pkgdesc="GigaAM v3 ASR backend for nerd-dictation (Russian speech recognition)"
arch=('x86_64')
url="https://github.com/nerd-voice-interface/nerd-dictation"
license=('MIT')
depends=('onnxruntime-gpu-sm61' 'gcc-libs' 'glibc' 'zstd')
makedepends=('git' 'cmake' 'python' 'python-onnx' 'python-numpy' 'zstd')
provides=('nerd-dictation-backend')
source=()
sha256sums=()

# Build from local source
_srcdir="${startdir}"

prepare() {
    cd "$_srcdir"

    # Initialize submodules if needed
    if [ ! -f "ThirdParty/kaldi-native-fbank/CMakeLists.txt" ]; then
        git submodule update --init --recursive
    fi
}

build() {
    cd "$_srcdir"

    # Out of tree, in makepkg's own build directory: the embedded model is
    # derived from the upstream volumes there (scripts/build_model.py).
    cmake -S . -B "$srcdir/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr \
        -DBUILD_SHARED_LIBS=ON

    cmake --build "$srcdir/build" -j$(nproc)
}

package() {
    cd "$_srcdir"

    # Install shared library (model and tokens are embedded)
    install -Dm755 "$srcdir/build/libnerd-dictation-gigaam.so.${pkgver}" \
        "${pkgdir}/usr/lib/libnerd-dictation-gigaam.so.${pkgver}"

    # Create symlinks
    ln -s libnerd-dictation-gigaam.so.${pkgver} \
        "${pkgdir}/usr/lib/libnerd-dictation-gigaam.so.3"
    ln -s libnerd-dictation-gigaam.so.${pkgver} \
        "${pkgdir}/usr/lib/libnerd-dictation-gigaam.so"

    # Install license
    install -Dm644 ThirdParty/nerd-dictation-api/LICENSE \
        "${pkgdir}/usr/share/licenses/${pkgname}/LICENSE"
}
