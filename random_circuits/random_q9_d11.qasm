OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
rz(2.5630753) q[3];
cx q[0],q[2];
cx q[6],q[5];
rz(6.0701356) q[1];
cx q[8],q[7];
rz(3.9929653) q[4];
cx q[1],q[3];
rz(0.10956243) q[5];
cx q[0],q[2];
cx q[6],q[8];
cx q[7],q[4];
cx q[0],q[2];
rz(4.4877547) q[7];
cx q[3],q[5];
rz(6.1528189) q[4];
cx q[6],q[1];
rz(3.2949021) q[8];
rz(5.1460813) q[3];
cx q[1],q[6];
cx q[0],q[8];
rz(5.2312809) q[7];
rz(0.42915484) q[4];
cx q[2],q[5];
cx q[3],q[0];
rz(3.2513482) q[2];
rz(1.5895018) q[6];
cx q[7],q[4];
rz(3.8479836) q[8];
cx q[5],q[1];
rz(5.7843752) q[1];
cx q[7],q[8];
rz(1.5556523) q[2];
rz(1.688741) q[5];
cx q[3],q[6];
rz(3.4198172) q[0];
rz(3.010018) q[4];
rz(2.1373234) q[0];
cx q[7],q[3];
cx q[4],q[8];
rz(5.4946404) q[2];
cx q[6],q[5];
rz(0.95612873) q[1];
rz(6.230932) q[0];
cx q[6],q[1];
rz(1.7666875) q[7];
cx q[8],q[2];
rz(5.9605782) q[5];
rz(0.24902782) q[3];
rz(1.5386142) q[4];
cx q[7],q[2];
rz(0.47460767) q[1];
rz(6.2717789) q[8];
rz(4.5808551) q[6];
cx q[4],q[0];
rz(3.6117282) q[3];
rz(4.6951415) q[5];
cx q[2],q[5];
rz(0.85688664) q[6];
rz(5.6470601) q[3];
cx q[7],q[4];
rz(4.6770349) q[8];
rz(3.8740507) q[0];
rz(2.0753936) q[1];
cx q[7],q[0];
cx q[1],q[2];
rz(3.6040629) q[6];
rz(1.5157545) q[5];
cx q[3],q[4];
rz(1.0844793) q[8];