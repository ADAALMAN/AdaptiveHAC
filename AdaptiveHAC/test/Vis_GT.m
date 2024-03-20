%load("./data/029_mon_Mix_Nic.mat")
tr = sig2timestamp(lbl_out,[1,14634],'nonzero');
imagesc(db(hil_resha_aligned(:,:,3)))
xline(tr,'r')