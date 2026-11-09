// api.js - tiny helpers to talk to the Python server.
// "fetch" is the browser's built-in way to send a web request.
async function handle(res) {
  if (!res.ok) {
    // the server sends {detail: "..."} when something is wrong; show that message
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch (e) { /* ignore */ }
    throw new Error(msg);
  }
  return res.json();
}
export const get = (url) => fetch(url).then(handle);
export const post = (url, body) =>
  fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then(handle);
export const postForm = (url, formData) => fetch(url, { method: "POST", body: formData }).then(handle);
