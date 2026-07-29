# Zaya Co-Host Dashboard

A single-page web dashboard for managing the Zaya Co-Host platform — voice packs, personas, usage analytics, API keys, and billing.

## Architecture

The dashboard is a **vanilla HTML/CSS/JS** single-page application with **no build step**. It's served as static files by the Jarvis server and communicates with the Jarvis API via `fetch()`.

```
┌─────────────────────────────────────┐
│  Browser                            │
│  ┌───────────────────────────────┐  │
│  │  index.html  ←─ shell + nav   │  │
│  │  style.css   ←─ dark theme    │  │
│  │  app.js      ←─ SPA logic     │  │
│  └──────────┬────────────────────┘  │
│             │ fetch()               │
└─────────────┼───────────────────────┘
              │
┌─────────────▼───────────────────────┐
│  Jarvis Server (port 8080)          │
│  ┌───────────────────────────────┐  │
│  │  /dashboard/  → static files  │  │
│  │  /v1/...      → JSON API      │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
```

## API Endpoints

The dashboard calls these endpoints on the Jarvis server:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Server health check |
| `/v1/voice/packs` | GET | List voice packs |
| `/v1/voice/clone` | POST | Clone voice from audio upload |
| `/v1/personas` | GET | List available personas |
| `/v1/persona` | GET | Get current persona config |
| `/v1/persona` | POST | Switch active persona |
| `/v1/persona/update` | POST | Update persona settings |
| `/v1/usage` | GET | Get usage stats for current owner |
| `/v1/api-key/create` | POST | Create new API key |
| `/v1/api-key/revoke` | POST | Revoke API key |
| `/v1/api-key/list` | GET | List API keys |
| `/v1/pricing` | GET | Get pricing information (public) |
| `/v1/billing/portal` | GET | Get billing portal URL |
| `/v1/chat/completions` | POST | Send a chat message (for test) |

## File Structure

```
site/dashboard/
├── index.html       # Main HTML shell with sidebar navigation
├── app.js           # SPA logic: router, API client, page components
├── style.css        # Dark theme CSS with responsive layout
├── app.test.js      # Standalone smoke test (Node.js)
└── README.md        # This file
```

## How to Add a New Page

1. **Add nav item** in `index.html`:
   ```html
   <button class="nav-item" data-page="my-page" onclick="navigate('#/my-page')">
     <span class="icon"><i class="fas fa-star"></i></span> My Page
   </button>
   ```

2. **Add route** in `app.js` `renderPage()`:
   ```javascript
   case 'my-page': renderMyPage(main); break;
   ```

3. **Write render function**:
   ```javascript
   async function renderMyPage(main) {
     main.innerHTML = `...page shell...`;
     const body = main.querySelector('.page-body');
     // Fetch data from API and populate
   }
   ```

4. **Register global** (for onclick handlers):
   ```javascript
   window.renderMyPage = renderMyPage;
   ```

## Adding API Endpoints

Add methods to the `API` client in `app.js`:

```javascript
// In the API object:
async put(path, body) { ... }

// Usage:
await API.put('/v1/my-endpoint', { key: 'value' });
```

## Authentication

- The API key is stored in `localStorage` under `zaya_api_key`
- All API calls include `Authorization: <key>` header if set
- Set/change the key in the **Settings** page

## Dark Theme

The dashboard uses the 1bit.systems brand colors:

| Role | Color | Hex |
|------|-------|-----|
| Background | Very dark navy | `#0a0a0f` |
| Surface | Dark purple-black | `#14141f` |
| Primary | Green | `#00ff88` |
| Secondary | Teal-blue | `#12a0ed` |
| Text | Light gray | `#e0e0e0` |
| Accent | Orange | `#ff9500` |
| Danger | Red | `#ff3355` |

## Dependencies (CDN)

- [Chart.js 4.4.1](https://www.chartjs.org/) — Usage graphs
- [Font Awesome 6.5.1](https://fontawesome.com/) — Icons

No other dependencies. No build step. No npm.

## Testing

Run the smoke test:

```bash
node site/dashboard/app.test.js
```

For full integration testing, start the Jarvis server and open the dashboard in a browser.

## Serving

The dashboard is served at `/dashboard/` by the Jarvis server (port 8080 by default):

```
http://localhost:8080/dashboard/
```

## License

MIT — part of the 1bit.systems open-source project.
