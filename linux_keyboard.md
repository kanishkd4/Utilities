```bash
kanishk@kanishk-MS-7D22:~$ cat /etc/X11/xorg.conf.d/00-keyboard.conf
Section "InputClass"
        Identifier "system-keyboard"
        MatchIsKeyboard "on"
        Option "XkbOptions" "ctrl:nocaps,altwin:swap_alt_win"
        Option "AutoRepeat" "250 60"
EndSection
```
