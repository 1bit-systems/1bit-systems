// 1bit-halo-helm-tray — Plasma StatusNotifierItem MVP.
//
// Registers a tray icon with `org.kde.StatusNotifierWatcher` over
// QtDBus. On Plasma this lights up in the system tray with our six
// menu items (Status / Start All / Stop All / Restart 1bit-server /
// Open 1bit.systems / Quit). Falls back to a regular tray on
// non-KDE systems via QSystemTrayIcon — Qt6 already negotiates the
// SNI bridge under the hood when the watcher is present.
//
// The pure logic (service set, state parser, status-line formatter)
// lives in onebit::helm::tray; this binary only does the wiring.

#include "onebit/helm/tray.hpp"

#include <QAction>
#include <QApplication>
#include <QIcon>
#include <QMenu>
#include <QSystemTrayIcon>
#include <QTimer>
#include <QtDBus/QDBusConnection>
#include <QtDBus/QDBusInterface>
#include <QtDBus/QDBusReply>

#include <cstdio>
#include <vector>

namespace tray = onebit::helm::tray;

namespace {

bool plasma_watcher_alive()
{
    auto bus = QDBusConnection::sessionBus();
    if (!bus.isConnected()) return false;
    QDBusInterface watcher("org.kde.StatusNotifierWatcher",
                           "/StatusNotifierWatcher",
                           "org.kde.StatusNotifierWatcher",
                           bus);
    return watcher.isValid();
}

void update_status_action(QAction* a,
                          const std::vector<tray::ServiceStatus>& rows)
{
    auto line = tray::build_status_line(rows);
    a->setText(QString("Status: ") + QString::fromStdString(line));
}

} // namespace

int main(int argc, char** argv)
{
    QApplication app(argc, argv);
    QApplication::setQuitOnLastWindowClosed(false);
    QApplication::setOrganizationName("1bit-systems");
    QApplication::setApplicationName("1bit-halo-helm-tray");

    if (!QSystemTrayIcon::isSystemTrayAvailable()) {
        std::fprintf(stderr,
                     "1bit-halo-helm-tray: no system tray available\n");
        return 1;
    }

    if (plasma_watcher_alive()) {
        std::fprintf(stderr,
            "1bit-halo-helm-tray: Plasma SNI watcher present, "
            "registering on org.kde.StatusNotifierWatcher\n");
    } else {
        std::fprintf(stderr,
            "1bit-halo-helm-tray: Plasma SNI watcher not found, "
            "falling back to legacy tray (Qt6 SNI bridge will pick "
            "up if the watcher appears later)\n");
    }

    QSystemTrayIcon icon;
    auto themed = QIcon::fromTheme(
        QString::fromUtf8(tray::ICON_THEME_NAME.data()));
    icon.setIcon(themed.isNull() ? QIcon() : themed);
    icon.setToolTip(QStringLiteral("1bit Helm"));

    QMenu menu;
    auto* status_action = menu.addAction(QStringLiteral("Status: …"));
    status_action->setEnabled(false);
    menu.addSeparator();
    auto* start_all = menu.addAction(
        QString::fromUtf8(tray::action_label(tray::Action::StartAll).data()));
    auto* stop_all  = menu.addAction(
        QString::fromUtf8(tray::action_label(tray::Action::StopAll).data()));
    auto* restart   = menu.addAction(
        QString::fromUtf8(tray::action_label(tray::Action::RestartServer).data()));
    auto* open_site = menu.addAction(
        QString::fromUtf8(tray::action_label(tray::Action::OpenSite).data()));
    menu.addSeparator();
    auto* quit_act  = menu.addAction(
        QString::fromUtf8(tray::action_label(tray::Action::Quit).data()));

    QObject::connect(start_all, &QAction::triggered, [] {
        std::vector<std::string_view> u(tray::SERVICES.begin(),
                                        tray::SERVICES.end());
        (void)tray::systemctl("start", u);
    });
    QObject::connect(stop_all, &QAction::triggered, [] {
        std::vector<std::string_view> u(tray::SERVICES.begin(),
                                        tray::SERVICES.end());
        (void)tray::systemctl("stop", u);
    });
    QObject::connect(restart, &QAction::triggered, [] {
        (void)tray::systemctl("restart",
                              std::vector<std::string_view>{"strix-server"});
    });
    QObject::connect(open_site, &QAction::triggered, [] {
        (void)tray::open_site();
    });
    QObject::connect(quit_act, &QAction::triggered, &QApplication::quit);

    icon.setContextMenu(&menu);
    icon.show();

    QTimer poll;
    poll.setInterval(tray::REFRESH_INTERVAL_MS);
    QObject::connect(&poll, &QTimer::timeout, [&] {
        auto rows = tray::probe_services();
        update_status_action(status_action, rows);
    });
    update_status_action(status_action, tray::probe_services());
    poll.start();

    return QApplication::exec();
}
