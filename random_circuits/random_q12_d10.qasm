OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
rz(3.7280826) q[4];
rz(5.1079386) q[7];
cx q[8],q[6];
cx q[11],q[0];
cx q[10],q[9];
rz(5.7815313) q[3];
rz(2.1567228) q[5];
rz(4.9160924) q[1];
rz(3.6053038) q[2];
rz(4.628382) q[8];
rz(2.6610103) q[7];
rz(0.090558418) q[10];
cx q[0],q[6];
rz(2.079429) q[11];
cx q[4],q[1];
cx q[5],q[3];
cx q[9],q[2];
cx q[11],q[4];
cx q[1],q[6];
cx q[0],q[8];
cx q[7],q[3];
rz(4.9461823) q[9];
rz(0.032389849) q[2];
rz(5.3227572) q[5];
rz(3.881776) q[10];
rz(2.5723158) q[5];
cx q[8],q[2];
cx q[9],q[0];
cx q[6],q[1];
rz(0.47359757) q[11];
rz(5.3563298) q[10];
rz(1.1994262) q[7];
cx q[3],q[4];
cx q[2],q[9];
rz(1.3103283) q[11];
rz(1.0241625) q[0];
cx q[3],q[8];
cx q[4],q[6];
rz(2.5020914) q[7];
rz(1.7706575) q[5];
cx q[1],q[10];
rz(4.0345613) q[2];
rz(3.9171767) q[1];
rz(3.5460872) q[8];
cx q[5],q[3];
rz(0.61412799) q[11];
cx q[4],q[7];
cx q[9],q[6];
cx q[0],q[10];
rz(3.7639656) q[3];
rz(2.833164) q[7];
cx q[2],q[6];
rz(3.8169107) q[0];
cx q[8],q[9];
rz(1.9279778) q[4];
cx q[5],q[10];
cx q[1],q[11];
rz(2.1090683) q[0];
cx q[11],q[10];
rz(1.7188774) q[3];
cx q[6],q[2];
rz(0.89913215) q[9];
rz(5.3285032) q[4];
cx q[1],q[5];
cx q[7],q[8];
rz(3.2115873) q[3];
cx q[8],q[0];
cx q[4],q[6];
cx q[7],q[9];
cx q[2],q[1];
cx q[10],q[11];
rz(4.3198824) q[5];
cx q[5],q[3];
rz(0.55467277) q[6];
cx q[1],q[8];
rz(1.1282828) q[9];
cx q[11],q[2];
cx q[7],q[4];
rz(5.6798698) q[10];
rz(2.0774271) q[0];
