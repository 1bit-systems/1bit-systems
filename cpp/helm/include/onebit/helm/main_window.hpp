// 1bit-helm — Qt6 Widgets MainWindow.
//
// Header is Qt-only — this is the part that *requires* Qt6 (and so
// is excluded from the pure-model test target). Window title +
// pane labels match the Rust eframe build bit-for-bit so muscle
// memory carries over.

#pragma once

#include "onebit/helm/app_model.hpp"
#include "onebit/helm/bearer.hpp"

#include <QMainWindow>
#include <memory>

class QStackedWidget;
class QListWidget;
class QListWidgetItem;
class QStatusBar;

namespace onebit::helm {

class StatusPane;
class ChatPane;
class ModelsPane;
class SettingsPane;

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(AppModel model, QWidget* parent = nullptr);
    ~MainWindow() override;

    // Apply a worker message — same path the eframe build's
    // `apply_ui_msg` runs. Public so the network worker thread can
    // marshal across with QMetaObject::invokeMethod.
    Q_INVOKABLE void onUiMessage(int kind, QString blob);

protected:
    void closeEvent(QCloseEvent* ev) override;

private slots:
    void onPaneSelected(int index);

private:
    void setupUi();
    void setupPanes();
    void persistSelectedPane();
    void restoreSelectedPane();

    AppModel        model_;
    Bearer          bearer_;
    QStackedWidget* stack_       {nullptr};
    QListWidget*    pane_list_   {nullptr};
    StatusPane*     status_pane_ {nullptr};
    ChatPane*       chat_pane_   {nullptr};
    ModelsPane*     models_pane_ {nullptr};
    SettingsPane*   settings_pane_{nullptr};
};

} // namespace onebit::helm
