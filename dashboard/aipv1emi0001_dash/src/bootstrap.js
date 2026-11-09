const startup = document.getElementById("dashboard-startup");
const message = document.getElementById("dashboard-startup-message");

import("./index.js").then(() => {
    startup.remove();
}).catch((error) => {
    startup.setAttribute("role", "alert");
    message.textContent = "Dashboard could not load. Check the UIBuilder browser libraries and reload.";
    console.error("Dashboard initialization failed:", error);
});
