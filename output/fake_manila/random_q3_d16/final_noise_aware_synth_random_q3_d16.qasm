OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
rz(4.9564665) q[3];
rz(4.1447778) q[2];
rz(3.5489867) q[2];
rz(3.0395156) q[3];
rz(3.5356476) q[4];
rz(4.1416544) q[4];
cx q[3],q[2];
cx q[4],q[3];
rz(0.83887464) q[3];
rz(5.1626532) q[3];
rz(5.6341808) q[3];
cx q[3],q[4];
rz(3.0230365) q[3];
cx q[2],q[3];
cx q[3],q[4];
rz(0.28863372) q[3];
rz(3.7237741) q[3];
rz(1.9304308) q[4];
rz(2.3334499) q[4];
rz(5.9945886) q[4];
rz(3.5807106) q[3];
rz(2.4429843) q[3];
rz(5.3987886) q[3];
rz(2.5358232) q[3];
rz(4.3981655) q[4];
rz(4.5933191) q[4];
rz(4.9920861) q[3];