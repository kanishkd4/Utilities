sudo nano /etc/X11/xorg.conf.d/99-apple-trackpad.conf

```bash
Section "InputClass"
    Identifier "Apple Magic Trackpad Configuration"
    MatchProduct "Apple Inc. Magic Trackpad"
    MatchIsTouchpad "on"
    Driver "libinput"
    
    # Set tracking speed (Adjust between -1.0 and 1.0)
    Option "AccelSpeed" "0.6"
    
    # Enable macOS-style two-finger physical click for Right-Click
    Option "ClickMethod" "clickfinger"
    
    # Enable tap-to-click
    Option "Tapping" "on"
    
    # Enable natural scrolling (reverse direction)
    Option "NaturalScrolling" "true"
EndSection
```
