OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
rz(4.9564665) q[1];
rz(4.1447778) q[2];
rz(3.5489867) q[2];
rz(3.0395156) q[1];
rz(3.5356476) q[0];
rz(4.1416544) q[0];
cx q[1],q[2];
cx q[0],q[1];
rz(0.83887464) q[1];
rz(5.1626532) q[1];
rz(5.6341808) q[1];
rz(3.0230365) q[1];
cx q[2],q[0];
cx q[2],q[1];
rz(0.28863372) q[1];
rz(3.7237741) q[1];
rz(1.9304308) q[0];
rz(2.3334499) q[0];
rz(5.9945886) q[0];
rz(3.5807106) q[1];
rz(2.4429843) q[1];
rz(5.3987886) q[1];
rz(2.5358232) q[1];
rz(4.3981655) q[0];
rz(4.5933191) q[0];
rz(4.9920861) q[1];
