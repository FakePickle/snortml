// ---------------- early-access form backend ----------------
// Where the sign-up form POSTs {email, role} as JSON. While url is empty the
// form says it is not connected instead of pretending to work.
//
// Supabase (table "waitlist" with columns email, role; insert-only RLS policy):
//   url: "https://<project-ref>.supabase.co/rest/v1/waitlist",
//   headers: { apikey: "<anon key>", Authorization: "Bearer <anon key>", Prefer: "return=minimal" },
//
// Formspree or any endpoint that accepts JSON:
//   url: "https://formspree.io/f/<form-id>",
//   headers: { Accept: "application/json" },
const WAITLIST = {
    url: "",
    headers: {},
};

// ---------------- example-alert tabs ----------------
document.querySelectorAll("[role=tablist]").forEach((list) => {
    const tabs = [...list.querySelectorAll("[role=tab]")];

    function select(tab, focus) {
        tabs.forEach((t) => {
            const on = t === tab;
            t.setAttribute("aria-selected", on);
            t.tabIndex = on ? 0 : -1;
            document
                .getElementById(t.getAttribute("aria-controls"))
                .classList.toggle("is-active", on);
        });
        if (focus) tab.focus();
    }

    tabs.forEach((tab, i) => {
        tab.addEventListener("click", () => select(tab));
        tab.addEventListener("keydown", (e) => {
            const step = { ArrowRight: 1, ArrowLeft: -1 }[e.key];
            if (!step) return;
            e.preventDefault();
            select(tabs[(i + step + tabs.length) % tabs.length], true);
        });
    });
});

// ---------------- early-access form ----------------
const form = document.getElementById("waitlist-form");

if (form) {
    const statusEl = document.getElementById("form-status");
    const submitBtn = document.getElementById("submit-btn");
    const submitText = submitBtn.textContent;

    function setStatus(text, kind) {
        statusEl.textContent = text;
        statusEl.className = "form-status" + (kind ? " " + kind : "");
    }

    if (!WAITLIST.url) {
        setStatus(
            "Sign-up is not connected yet: set WAITLIST.url in site.js.",
            "error",
        );
    }

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        if (!WAITLIST.url) return;

        const email = form.elements.email.value.trim();
        const role = form.elements.role.value || null;
        if (!email) return;

        submitBtn.disabled = true;
        submitBtn.textContent = "Adding...";
        setStatus("");

        try {
            const res = await fetch(WAITLIST.url, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    ...WAITLIST.headers,
                },
                body: JSON.stringify({ email, role }),
            });

            if (res.ok) {
                setStatus(
                    "✓ You're on the list. We'll email you when early access opens.",
                    "success",
                );
                form.reset();
            } else if (res.status === 409) {
                // unique-constraint violation on the email column
                setStatus("That email is already on the list.", "success");
            } else {
                throw new Error("request failed: " + res.status);
            }
        } catch (err) {
            setStatus(
                "Something went wrong and you were not added. Try again in a moment.",
                "error",
            );
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = submitText;
        }
    });
}
