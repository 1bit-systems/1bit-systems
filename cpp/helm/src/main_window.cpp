#include "onebit/helm/main_window.hpp"

#include "onebit/helm/bearer.hpp"
#include "onebit/helm/conv_log.hpp"

#include <QAction>
#include <QApplication>
#include <QCloseEvent>
#include <QFrame>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QInputDialog>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QProgressBar>
#include <QPushButton>
#include <QScrollArea>
#include <QSettings>
#include <QStackedWidget>
#include <QStatusBar>
#include <QTextEdit>
#include <QVBoxLayout>
#include <QWidget>

#include <utility>

namespace onebit::helm {

namespace {

// QSettings key for persisting the currently-selected pane index.
constexpr const char* kSelectedPaneKey = "helm/current_pane";
constexpr const char* kOrgName         = "1bit-systems";
constexpr const char* kAppName         = "1bit-helm";

QString brand_qstr()
{
    return QString::fromUtf8(BRAND.data(),
                             static_cast<int>(BRAND.size()));
}
QString brand_domain_qstr()
{
    return QString::fromUtf8(BRAND_DOMAIN.data(),
                             static_cast<int>(BRAND_DOMAIN.size()));
}

} // namespace

// ---------------------------------------------------------------------
// Pane widgets
// ---------------------------------------------------------------------

class StatusPane : public QWidget {
public:
    explicit StatusPane(AppModel& m, QWidget* parent = nullptr)
        : QWidget(parent), model_(m)
    {
        auto* root = new QVBoxLayout(this);
        auto* heading = new QLabel(QStringLiteral("<h2>Status</h2>"));
        root->addWidget(heading);

        endpoint_ = new QLabel(this);
        endpoint_->setTextFormat(Qt::PlainText);
        root->addWidget(endpoint_);

        auto* grid_holder = new QFrame(this);
        auto* grid        = new QGridLayout(grid_holder);
        int row = 0;
        grid->addWidget(new QLabel("loaded model"), row, 0);
        loaded_model_ = new QLabel("—");
        grid->addWidget(loaded_model_, row++, 1);

        grid->addWidget(new QLabel("tok/s (decode)"), row, 0);
        tok_s_ = new QProgressBar();
        tok_s_->setRange(0, 120);
        grid->addWidget(tok_s_, row++, 1);

        grid->addWidget(new QLabel("GPU temp"), row, 0);
        gpu_temp_ = new QLabel("—");
        grid->addWidget(gpu_temp_, row++, 1);

        grid->addWidget(new QLabel("GPU util"), row, 0);
        gpu_util_ = new QProgressBar();
        gpu_util_->setRange(0, 100);
        grid->addWidget(gpu_util_, row++, 1);

        grid->addWidget(new QLabel("NPU"), row, 0);
        npu_ = new QLabel("down");
        grid->addWidget(npu_, row++, 1);

        grid->addWidget(new QLabel("shadow-burn %"), row, 0);
        shadow_ = new QLabel("—");
        grid->addWidget(shadow_, row++, 1);

        grid->addWidget(new QLabel("stale"), row, 0);
        stale_ = new QLabel("no");
        grid->addWidget(stale_, row++, 1);
        root->addWidget(grid_holder);

        services_ = new QLabel("(no service probes yet — waiting for "
                               "first telemetry tick)");
        services_->setWordWrap(true);
        root->addWidget(services_);
        root->addStretch();
        refresh();
    }

    void refresh()
    {
        endpoint_->setText(QString::fromStdString(
            "telemetry: " + model_.landing_url + "/_live/stats"));
        const auto& s = model_.live;
        loaded_model_->setText(s.loaded_model.empty()
                                   ? QStringLiteral("—")
                                   : QString::fromStdString(s.loaded_model));
        tok_s_->setValue(static_cast<int>(s.tok_s_decode));
        tok_s_->setFormat(QString::number(s.tok_s_decode, 'f', 1));
        gpu_temp_->setText(QString::number(static_cast<int>(s.gpu_temp_c))
                           + " °C");
        gpu_util_->setValue(static_cast<int>(s.gpu_util_pct));
        npu_->setText(s.npu_up ? "up" : "down");
        shadow_->setText(QString::number(s.shadow_burn_exact_pct, 'f', 1));
        stale_->setText(s.stale ? "yes" : "no");
        if (s.services.empty()) {
            services_->setText("(no service probes yet)");
        } else {
            QString line;
            for (const auto& svc : s.services) {
                if (!line.isEmpty()) line += "  ·  ";
                line += QString::fromStdString(svc.name);
                line += svc.active ? " (up)" : " (down)";
            }
            services_->setText(line);
        }
    }

private:
    AppModel&     model_;
    QLabel*       endpoint_     {};
    QLabel*       loaded_model_ {};
    QProgressBar* tok_s_        {};
    QLabel*       gpu_temp_     {};
    QProgressBar* gpu_util_     {};
    QLabel*       npu_          {};
    QLabel*       shadow_       {};
    QLabel*       stale_        {};
    QLabel*       services_     {};
};

class ChatPane : public QWidget {
public:
    explicit ChatPane(AppModel& m, QWidget* parent = nullptr)
        : QWidget(parent), model_(m)
    {
        auto* root = new QVBoxLayout(this);
        root->addWidget(new QLabel(QStringLiteral("<h2>Chat</h2>")));

        log_view_ = new QTextEdit();
        log_view_->setReadOnly(true);
        root->addWidget(log_view_, 1);

        auto* row = new QHBoxLayout();
        input_ = new QLineEdit();
        input_->setPlaceholderText("message…");
        row->addWidget(input_, 1);
        send_ = new QPushButton("Send");
        row->addWidget(send_);
        root->addLayout(row);

        QObject::connect(send_, &QPushButton::clicked,
                         this, &ChatPane::onSend);
        QObject::connect(input_, &QLineEdit::returnPressed,
                         this, &ChatPane::onSend);
        refresh();
    }

    void refresh()
    {
        QString out;
        for (const auto& t : model_.chat_conv.turns()) {
            if (t.role == Role::System) continue;
            const char* who = (t.role == Role::User) ? "you" : "halo";
            out += QStringLiteral("<b>%1:</b> ").arg(who);
            out += QString::fromStdString(t.content).toHtmlEscaped();
            out += "<br><br>";
        }
        if (model_.chat_streaming.has_value()) {
            out += QStringLiteral("<b>halo:</b> ");
            out += QString::fromStdString(*model_.chat_streaming).toHtmlEscaped();
        }
        log_view_->setHtml(out);
    }

private:
    void onSend()
    {
        auto txt = input_->text().trimmed();
        if (txt.isEmpty()) return;
        model_.chat_conv.push_user(txt.toStdString());
        input_->clear();
        // The live network worker is wired in main.cpp's MainWindow;
        // the pane only mutates the model + signals refresh.
        refresh();
    }

    AppModel&  model_;
    QTextEdit* log_view_ {};
    QLineEdit* input_    {};
    QPushButton* send_   {};
};

class ModelsPane : public QWidget {
public:
    explicit ModelsPane(AppModel& m, QWidget* parent = nullptr)
        : QWidget(parent), model_(m)
    {
        auto* root = new QVBoxLayout(this);
        auto* row  = new QHBoxLayout();
        root->addWidget(new QLabel(QStringLiteral("<h2>Models</h2>")));
        refresh_ = new QPushButton("Refresh");
        row->addWidget(refresh_);
        count_lbl_ = new QLabel();
        row->addWidget(count_lbl_, 1);
        root->addLayout(row);

        list_ = new QListWidget();
        root->addWidget(list_, 1);
        QObject::connect(refresh_, &QPushButton::clicked,
                         this, &ModelsPane::refresh);
        refresh();
    }

    void refresh()
    {
        list_->clear();
        if (model_.models_error.has_value()) {
            list_->addItem("error: "
                           + QString::fromStdString(*model_.models_error));
        }
        if (model_.models.empty() && !model_.models_error.has_value()) {
            list_->addItem(
                "(no models loaded — click Refresh, or the gateway may be down)");
        }
        for (const auto& c : model_.models) {
            QString line = QString::fromStdString(c.id);
            if (!c.owned_by.empty()) {
                line += "   owned_by: ";
                line += QString::fromStdString(c.owned_by);
            }
            list_->addItem(line);
        }
        count_lbl_->setText(
            QString("%1 model(s) @ %2")
                .arg(model_.models.size())
                .arg(QString::fromStdString(model_.gateway_url)));
    }

private:
    AppModel&    model_;
    QPushButton* refresh_  {};
    QLabel*      count_lbl_{};
    QListWidget* list_     {};
};

class SettingsPane : public QWidget {
public:
    explicit SettingsPane(AppModel& m, Bearer& bearer, QWidget* parent = nullptr)
        : QWidget(parent), model_(m), bearer_(bearer)
    {
        auto* root = new QVBoxLayout(this);
        root->addWidget(new QLabel(QStringLiteral("<h2>Settings</h2>")));

        backend_ = new QLabel();
        root->addWidget(backend_);
        status_ = new QLabel();
        root->addWidget(status_);

        auto* row = new QHBoxLayout();
        edit_ = new QLineEdit();
        edit_->setEchoMode(QLineEdit::Password);
        edit_->setPlaceholderText("paste bearer …");
        row->addWidget(edit_, 1);
        save_ = new QPushButton("Save");
        reset_ = new QPushButton("Reset");
        row->addWidget(save_);
        row->addWidget(reset_);
        root->addLayout(row);

        endpoints_ = new QLabel();
        root->addWidget(endpoints_);

        about_ = new QLabel();
        about_->setTextInteractionFlags(Qt::TextBrowserInteraction);
        root->addWidget(about_);
        root->addStretch();

        QObject::connect(save_,  &QPushButton::clicked,
                         this, &SettingsPane::onSave);
        QObject::connect(reset_, &QPushButton::clicked,
                         this, &SettingsPane::onReset);
        refresh();
    }

    void refresh()
    {
        backend_->setText(
            QString("backend: ")
            + QString::fromUtf8(
                bearer_backend_label(bearer_.backend()).data()));
        status_->setText(QString("status: ")
                         + (bearer_.get().has_value() ? "set" : "unset"));
        endpoints_->setText(
            QString("gateway: %1\nlanding: %2")
                .arg(QString::fromStdString(model_.gateway_url))
                .arg(QString::fromStdString(model_.landing_url)));
        about_->setText(
            QString("%1 — %2\n1bit-helm v0.1.0 — Qt6 Widgets desktop client")
                .arg(brand_qstr(), brand_domain_qstr()));
    }

private:
    void onSave()
    {
        auto txt = edit_->text().trimmed().toStdString();
        if (txt.empty()) return;
        auto rc = bearer_.store(txt);
        if (!rc) {
            QMessageBox::warning(this, "Bearer",
                QString::fromStdString("bearer: " + rc.error()));
            return;
        }
        model_.cfg.bearer = bearer_.get();
        model_.toast      = "bearer stored";
        edit_->clear();
        refresh();
    }
    void onReset()
    {
        auto rc = bearer_.clear();
        if (!rc) {
            QMessageBox::warning(this, "Bearer",
                QString::fromStdString("bearer clear: " + rc.error()));
            return;
        }
        model_.cfg.bearer = std::nullopt;
        model_.toast      = "bearer cleared";
        refresh();
    }

    AppModel&    model_;
    Bearer&      bearer_;
    QLabel*      backend_   {};
    QLabel*      status_    {};
    QLineEdit*   edit_      {};
    QPushButton* save_      {};
    QPushButton* reset_     {};
    QLabel*      endpoints_ {};
    QLabel*      about_     {};
};

// ---------------------------------------------------------------------
// MainWindow
// ---------------------------------------------------------------------

MainWindow::MainWindow(AppModel model, QWidget* parent)
    : QMainWindow(parent), model_(std::move(model))
{
    setWindowTitle(brand_qstr() + QStringLiteral(" — helm"));
    resize(1024, 720);
    setupUi();
    setupPanes();
    restoreSelectedPane();
}

MainWindow::~MainWindow() = default;

void MainWindow::setupUi()
{
    pane_list_ = new QListWidget(this);
    pane_list_->setMaximumWidth(180);
    for (auto p : PANES_ALL) {
        pane_list_->addItem(QString::fromUtf8(pane_label(p).data()));
    }
    QObject::connect(pane_list_,
                     &QListWidget::currentRowChanged,
                     this, &MainWindow::onPaneSelected);

    stack_ = new QStackedWidget(this);

    auto* central = new QWidget(this);
    auto* h       = new QHBoxLayout(central);
    h->addWidget(pane_list_);
    h->addWidget(stack_, 1);
    setCentralWidget(central);

    statusBar()->showMessage(brand_domain_qstr());
}

void MainWindow::setupPanes()
{
    bearer_.load();

    status_pane_   = new StatusPane(model_, this);
    chat_pane_     = new ChatPane(model_, this);
    models_pane_   = new ModelsPane(model_, this);
    settings_pane_ = new SettingsPane(model_, bearer_, this);

    stack_->addWidget(status_pane_);
    stack_->addWidget(chat_pane_);
    stack_->addWidget(models_pane_);
    stack_->addWidget(settings_pane_);
}

void MainWindow::onPaneSelected(int index)
{
    if (index < 0 || index >= static_cast<int>(std::size(PANES_ALL))) return;
    model_.current_pane = PANES_ALL[index];
    stack_->setCurrentIndex(index);
    persistSelectedPane();
}

void MainWindow::persistSelectedPane()
{
    QSettings settings(kOrgName, kAppName);
    settings.setValue(kSelectedPaneKey,
                      QString::fromUtf8(pane_label(model_.current_pane).data()));
}

void MainWindow::restoreSelectedPane()
{
    QSettings settings(kOrgName, kAppName);
    auto      label = settings.value(kSelectedPaneKey).toString().toStdString();
    auto      p     = pane_from_string(label).value_or(Pane::Status);
    model_.current_pane = p;
    int idx = 0;
    for (std::size_t i = 0; i < std::size(PANES_ALL); ++i) {
        if (PANES_ALL[i] == p) { idx = static_cast<int>(i); break; }
    }
    pane_list_->setCurrentRow(idx);
    stack_->setCurrentIndex(idx);
}

void MainWindow::onUiMessage(int kind, QString blob)
{
    // Hook for the future network worker — kind/blob are placeholder
    // payload until the live SSE pump lands. Keeping the slot in place
    // so the QMetaObject::invokeMethod path is type-stable.
    Q_UNUSED(kind);
    Q_UNUSED(blob);
}

void MainWindow::closeEvent(QCloseEvent* ev)
{
    persistSelectedPane();
    if (auto rc = flush_conversation(model_); !rc) {
        // Log to stderr but never block close.
        std::fprintf(stderr, "helm: conv flush failed: %s\n",
                     rc.error().c_str());
    }
    QMainWindow::closeEvent(ev);
}

} // namespace onebit::helm
