// 1bit-helm — Qt6 Widgets desktop client against lemond
// (the C++ /v1/* gateway at /home/bcloud/repos/lemonade/).
//
// Renamed from halo-gaia 2026-04-20. Same env-var contract the Rust
// build honoured so existing shell profiles roll forward without a
// rewrite.
//
// Window title + pane labels match the Rust eframe build verbatim.

#include "onebit/helm/app_model.hpp"
#include "onebit/helm/main_window.hpp"

#include <QApplication>

int main(int argc, char** argv)
{
    QApplication app(argc, argv);
    QApplication::setOrganizationName("1bit-systems");
    QApplication::setApplicationName("1bit-helm");
    QApplication::setApplicationDisplayName(QStringLiteral("1bit monster — helm"));

    auto cfg = onebit::helm::load_config_from_env();
    auto model = onebit::helm::make_app_model(std::move(cfg));
    model.landing_url = onebit::helm::load_landing_url_from_env();

    onebit::helm::MainWindow win(std::move(model));
    win.show();
    return QApplication::exec();
}
