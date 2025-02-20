Unzip the archive for your operating system and copy that directory to the `plugins` directory of your Xournal++ installation (*Shared resources folder* or *Config folder*[^XOURNALPP_INSTALLATION_FOLDER]).

[^XOURNALPP_INSTALLATION_FOLDER]: https://xournalpp.github.io/guide/plugins/plugins/#installation-folder

To get the plugin custom icon to show up when adding the toolbar icon for a plugin, copy the `.svg` file in the plugin directory to a GTK supported icon location (e.g. `$HOME/.local/share/icons/` or `/usr/share/icons/` on Linux and `%LOCALAPPDATA%\icons\` or `%PROGRAM_FILES%\Xournal++\share\icons\` on Windows).

**Windows:**

If the plugin directory contains `.dll` files this additionally requires adding the directory to the `PATH` environment variable so that the contained `.dll` files can be found when loading main plugin `.dll` file from Lua in Xournal++.

To do this search for *Edit Environment variables for your account* in the Windows search, click *Environment Variables*, on the upper list of the User variables click the existing entry `Path` and add a new directory path of the plugin directory (e.g. `%LOCALAPPDATA%\xournalpp\plugins\PLUGIN_NAME`).
After applying the changes Xournal++ needs to be restarted.
