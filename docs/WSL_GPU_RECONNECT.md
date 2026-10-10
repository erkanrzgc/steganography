# Reconnect the existing own-GPU endpoint

Run only in the existing Ubuntu-24.04 WSL terminal as `erkanrzgc`, not
PowerShell or Kali. The earlier dedicated key must already exist. This recreates
only the prior loopback-only, public-key-only endpoint for four hours. No public
listener, password SSH authentication or permanent sshd configuration change.
Do not rerun the original bootstrap: it correctly refuses to overwrite the key.

```bash
test "$(id -un)" = erkanrzgc &&
test -f /home/erkanrzgc/.ssh/steganography_gpu_20261009_authorized_keys &&
test ! -L /home/erkanrzgc/.ssh &&
test ! -L /home/erkanrzgc/.ssh/steganography_gpu_20261009_authorized_keys &&
sudo install -d -m 0755 /run/sshd &&
sudo systemd-run --unit=steganography-gpu-ssh-batch-20261010 \
  --property=RuntimeMaxSec=4h \
  /usr/sbin/sshd -D -e -f /dev/null -p 22222 \
  -o ListenAddress=127.0.0.1 \
  -o HostKey=/etc/ssh/ssh_host_ed25519_key \
  -o PubkeyAuthentication=yes -o PasswordAuthentication=no \
  -o KbdInteractiveAuthentication=no -o PermitRootLogin=no \
  -o DisableForwarding=yes -o 'Subsystem=sftp internal-sftp' \
  -o AllowUsers=erkanrzgc \
  -o AuthorizedKeysFile=/home/erkanrzgc/.ssh/steganography_gpu_20261009_authorized_keys
```

If the dedicated unit is already active, do not start another server. Inspect
`systemctl status steganography-gpu-ssh-batch-20261010` instead. A collision on
port 22222 means inspect the existing listener, not kill an unrelated process.

The reverse listener on Kali port 22240 currently exists. First ask the assistant
to re-check connectivity after starting the server; no duplicate tunnel is
needed if it works. Only if that listener has disappeared, use the previously
verified tunnel and leave the terminal open:

```bash
ssh -N -T -a -x -o ConnectTimeout=15 \
  -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -o StrictHostKeyChecking=yes \
  -R 127.0.0.1:22240:127.0.0.1:22222 erkanrzgc@192.168.6.128
```

Never disable host-key verification or share a password/private key. Closing
the tunnel stops access; the dedicated server expires after four hours.
