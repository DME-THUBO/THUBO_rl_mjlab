# THUBO I24 — new body archive

**Branch: `sxd/new-THUBO`. Status: archived design proposal, not the active robot.**

新的本体 I24 保存在本分支。后续仍使用原有本体进行控制、起身和行走策略开发；有新的经费、购买新的舵机后，再继续新本体开发。

새 I24 본체는 이 브랜치에 보관한다. 이후 제어·보행·증류 개발은 기존 본체로 계속하며, 새 예산과 서보가 준비되면 새 본체 개발을 재개한다.

- [한국어 제작 안내](README_KO.md), [이미지 안내](START_HERE.html).
- [New STL](STL_NEW): 11 unique files / 13 printed instances. L34/R34 each require two copies.
- [Existing feet](STL_REUSED): unchanged original L6/R6 files.
- [A1 projects](A1_PROJECTS): 11 production plates plus optional F7/F8 fit coupons; each L34/R34 plate already contains two copies.
- [Editable CAD](CAD): individual STEP files and **printed-parts-only** assembly.
- [Checks](CHECKS): original local CAD/slicing checks, with the physical-validation limits below.

The supplied motor and official Orange Pi 5 reference solids are not redistributed here. The local full reference assembly (311 solids) was checked separately; this archive's assembly contains only the 15 printed robot parts. This branch does not replace `mjcf/THUBO.xml`, `meshes/`, task defaults or trained policies with I24.

Original inputs: motor SHA-256 `4182ac210720ac5d0a0724aaf29d0d5fbfa58c98439a1431cbaabb80ac6ac201`; Orange Pi 5 SHA-256 `efd859cdb210f850ee92a8cd427a59c32576f7c74a7fca5fe080f3b5c00e3070`.

Limitations: sample CAD collision checks and slicing passed. Actual motor identity/thread depth, fastening, PLA strength/fatigue, complete harness/cooler fit and walking remain unverified. The motor file's internal name ST-3235M differs from its ST-3250M filename. First fit-test F7 and the candidate screw lengths. See the guide for the approximately 1.52 mm minimum inner web beside the rear screw hole.

Commit author email requested by the owner: `gobdungi@gmail.com`.
