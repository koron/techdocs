## 無効化してみたユニット

* nfs-client.target
* avahi-daemon.service
* avahi-daemon.socket
* keyboard-setup.service (キーボードロケールの設定)

## 現状生きてるユニットの一覧

```console
$ systemctl list-unit-files | grep enabled | grep -v disabled | grep -v masked | grep -v indirect
bluetooth.service                          enabled         enabled
console-setup.service                      enabled         enabled
cron.service                               enabled         enabled
dhcpcd.service                             enabled         enabled
dphys-swapfile.service                     enabled         enabled
e2scrub_reap.service                       enabled         enabled
fake-hwclock.service                       enabled         enabled
getty@.service                             enabled         enabled
hciuart.service                            enabled         enabled
ModemManager.service                       enabled         enabled
networking.service                         enabled         enabled
raspberrypi-net-mods.service               enabled         enabled
rc-local.service                           enabled-runtime enabled
rpi-display-backlight.service              enabled         enabled
rpi-eeprom-update.service                  enabled         enabled
rsync.service                              enabled         enabled
rsyslog.service                            enabled         enabled
ssh.service                                enabled         enabled
sshswitch.service                          enabled         enabled
systemd-fsck-root.service                  enabled-runtime enabled
systemd-pstore.service                     enabled         enabled
systemd-remount-fs.service                 enabled-runtime enabled
systemd-timesyncd.service                  enabled         enabled
triggerhappy.service                       enabled         enabled
udisks2.service                            enabled         enabled
wpa_supplicant.service                     enabled         enabled
triggerhappy.socket                        enabled         enabled
remote-fs.target                           enabled         enabled
apt-daily-upgrade.timer                    enabled         enabled
apt-daily.timer                            enabled         enabled
e2scrub_all.timer                          enabled         enabled
fstrim.timer                               enabled         enabled
logrotate.timer                            enabled         enabled
man-db.timer                               enabled         enabled
```

## さらに無効化したユニット

* apt-daily-upgrade.timer
* apt-daily.timer
* man-db.timer (manデータベースの更新)

```console
$ systemctl list-unit-files | grep enabled | grep -v disabled | grep -v masked | grep -v indirect | grep -v .target
bluetooth.service                          enabled         enabled
console-setup.service                      enabled         enabled
cron.service                               enabled         enabled
dhcpcd.service                             enabled         enabled
dphys-swapfile.service                     enabled         enabled
e2scrub_reap.service                       enabled         enabled
fake-hwclock.service                       enabled         enabled
getty@.service                             enabled         enabled
hciuart.service                            enabled         enabled
ModemManager.service                       enabled         enabled
networking.service                         enabled         enabled
raspberrypi-net-mods.service               enabled         enabled
rc-local.service                           enabled-runtime enabled
rpi-display-backlight.service              enabled         enabled
rpi-eeprom-update.service                  enabled         enabled
rsync.service                              enabled         enabled
rsyslog.service                            enabled         enabled
ssh.service                                enabled         enabled
sshswitch.service                          enabled         enabled
systemd-fsck-root.service                  enabled-runtime enabled
systemd-pstore.service                     enabled         enabled
systemd-remount-fs.service                 enabled-runtime enabled
systemd-timesyncd.service                  enabled         enabled
triggerhappy.service                       enabled         enabled
udisks2.service                            enabled         enabled
wpa_supplicant.service                     enabled         enabled
triggerhappy.socket                        enabled         enabled
e2scrub_all.timer                          enabled         enabled
fstrim.timer                               enabled         enabled
logrotate.timer                            enabled         enabled
```

## さらに無効化したユニット2

* triggerhappy.service (キーボードショートカットらしい)
* triggerhappy.socket (キーボードショートカットらしい)
* ModemManager.service
* sshswitch.service (/boot/sshファイルがあればsshdを1回だけ起動する。/boot/sshは消すので1回だけ)
* udisks2.service
* console-setup.service
* alsa-restore.service
