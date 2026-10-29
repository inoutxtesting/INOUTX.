"use strict";

(function () {
  const configAdminEmail = "inoutx.testing@gmail.com";
  const adminSettingsPage =
    /(?:^|\/)system-config\.html$/.test(window.location.pathname);

  if (adminSettingsPage) {
    document.documentElement.style.visibility = "hidden";
  }

  async function checkAdminAccess() {
    const response = await fetch("/api/config-access", {
      method: "GET",
      credentials: "include",
      cache: "no-store",
      headers: {
        Accept: "application/json"
      }
    });

    if (response.status === 401 || response.status === 403) {
      return false;
    }

    if (!response.ok) {
      throw new Error(
        "Admin access check failed with HTTP " + response.status
      );
    }

    const result = await response.json();

    return result.success === true &&
      result.authorized === true &&
      String(result.email || "").trim().toLowerCase() === configAdminEmail &&
      String(result.role || "").trim().toLowerCase() === "config_admin";
  }

  const accessCheck = checkAdminAccess();

  document.addEventListener("DOMContentLoaded", function () {
    const adminNavLinks =
      document.querySelectorAll("[data-admin-only]");

    if (!adminNavLinks.length) {
      return;
    }

    accessCheck.then(function (authorized) {
      adminNavLinks.forEach(function (adminNavLink) {
        adminNavLink.style.display = authorized ? "" : "none";
        adminNavLink.setAttribute("aria-hidden", String(!authorized));
        adminNavLink.tabIndex = authorized ? 0 : -1;
      });
    }).catch(function (error) {
      adminNavLinks.forEach(function (adminNavLink) {
        adminNavLink.style.display = "none";
        adminNavLink.setAttribute("aria-hidden", "true");
        adminNavLink.tabIndex = -1;
      });

    });
  });

  if (adminSettingsPage) {
    accessCheck.then(function (authorized) {
      if (!authorized) {
        window.location.replace("/html/index.html");
        return;
      }

      document.documentElement.style.visibility = "";
    }).catch(function (error) {

      window.location.replace("/html/index.html");
    });
  }
})();
