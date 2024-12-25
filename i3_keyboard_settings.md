vim ~/bin/keyboard_settings

```
#!/bin/bash
# Restore keyboard settings
setxkbmap -option "ctrl:nocaps" -option "altwin:swap_alt_win"
xset r rate 250 60
```

sudo nano /etc/systemd/system/keyboard-resume.service

```
[Unit]
Description=Restore keyboard settings after suspend
After=suspend.target

[Service]
User=kanishk
Type=simple
Environment=DISPLAY=:0
ExecStart=/home/kanishk/bin/keyboard_settings

[Install]
WantedBy=suspend.target
```

sudo chmod 644 /etc/systemd/system/keyboard-resume.service
sudo systemctl daemon-reload
sudo systemctl enable keyboard-resume.service
sudo systemctl start keyboard-resume.service
chmod +x ~/bin/keyboard_settings


<!-- It's not perfect but it takes effect if I restart i3 using mod + shift + r -->