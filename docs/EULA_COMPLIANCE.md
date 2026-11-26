# Two-Stage Asset Installation & Minecraft EULA Compliance

`Solara-Ground` is a 100% clean-room native Rust client licensed under **GPL-3.0-only**:

1. **Zero Mojang Asset Redistribution:** No Mojang `.jar`, `.png`, `.ogg`, models, or fonts are committed or hosted in this repository or on Solara's servers.
2. **Official Mojang CDN Delivery via Solara Launcher:** During installation, **Solara Launcher** authenticates the user's paid Microsoft/Minecraft account and downloads official game assets directly from Mojang's CDN onto the user's machine (`--assets-dir`), while downloading only Solara's clean-room Rust binary from Solara's release servers.

> NOT AN OFFICIAL MINECRAFT PRODUCT. NOT APPROVED BY OR ASSOCIATED WITH MOJANG OR MICROSOFT.
