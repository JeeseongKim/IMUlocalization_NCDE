from loss import *

torch.multiprocessing.set_start_method('spawn', force=True)

from model.NODE import *
from model import MyCDE_
import utils_

def M9_train_pose_CDE_P(**kwargs):
    print("--Training M9 CDE!!--")
    ## predict position
    num_epochs = 1000
    batch_size = 1024
    max_norm = 1.0

    window = 100

    scale = 1.0

    vis = visdom.Visdom()

    AllLoss = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='All loss(CDE_P)'))

    plotcidst = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Cdist(CDE_P)'))
    plotL2 = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='L2(CDE_P)'))
    plotL1 = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='L1(CDE_P)'))
    plotCosim = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Cosim(CDE_P)'))
    plotKL = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='KL(CDE_P)'))
    Validation_plot = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Validation Loss avg(CDE_P)'))

    cde_model = MyCDE_.model_cde_P(input_channels=6, hidden_channels=128, output_channels=2, interpolation="cubic", **kwargs).cuda()
    cde_model = nn.DataParallel(cde_model).cuda()
    #optimizer_AdamW_cde = torch.optim.AdamW(cde_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer_AdamW_cde = torch.optim.Adam(cde_model.parameters(), lr=1e-3, weight_decay=1e-4)
    #optimizer_AdamW_cde = torch.optim.SGD(cde_model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-3)
    lr_scheduler_cde = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_AdamW_cde, 'min', factor=0.5, patience=5, verbose=-1, threshold=1e-6, min_lr=1e-7)

    # load ckpt
    load_ckpt = "/home/jsk/IMUlocalization/ckpt/rrandom.pth"
    if os.path.exists(load_ckpt):
        print("-----Loading Checkpoint-----")
        checkpoint = torch.load(load_ckpt)

        save_epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        # IMUTransformer.module.load_state_dict(checkpoint['IMUTransformer'])
        cde_model.load_state_dict(checkpoint['cde_model'])
        #optimizer_AdamW_cde.load_state_dict(checkpoint['optimizer_AdamW_cde'])
        lr_scheduler_cde.load_state_dict(checkpoint['lr_scheduler_cde'])

    dataset_load_start = time.time()
    print("---Loading Dataset---")
    MYdataset = utils_.my_dataset_gyroz_M9(window_size=window)
    train_loader = DataLoader(dataset=MYdataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    imu_file = "/home/jsk/IMUlocalization/M9/test/imu/synced_M9_sq_01.txt"
    vicon_file = "/home/jsk/IMUlocalization/M9/test/vicon/synced_vicon_sq_01.txt"

    MYValdataset = utils_.my_dataset_gyroz_M9_test(window_size=window, imu_filename=imu_file, vicon_filename=vicon_file)
    validation_loader = DataLoader(dataset=MYValdataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    print("Dataset Loading time: ", time.time()-dataset_load_start)
    if os.path.exists(load_ckpt):
        print("!!")
        start_epoch = save_epoch
        #start_epoch = 0
        print(start_epoch)
    else:
        start_epoch = 0

    start_epoch = 0
    for epoch in tqdm(range(start_epoch, num_epochs, 1)):
        print("start_epoch: ", start_epoch)
        print("EPOCH: ", epoch)
        running_loss = 0
        running_l2_loss = 0
        running_cdist_loss = 0
        running_l1_loss = 0
        running_cosim_loss = 0
        running_kl_loss = 0

        for i, data in enumerate(tqdm(train_loader)):
            data_m9, data_vicon = data
            t_imu = data_m9[:, :, 0].unsqueeze(2).cuda()
            #angle_z = data_m9[:, :, 1].unsqueeze(2).cuda()
            acc_data = data_m9[:, :, 2:4].cuda()
            gyro_z_imu = data_m9[:, :, 4].unsqueeze(2).cuda()
            ofs = data_m9[:, :, 5:7].cuda()
            cde_src = torch.cat([t_imu, acc_data, gyro_z_imu, ofs], dim=2).cuda()
            pose_xy = data_vicon[:, :, 1:3].cuda() * scale #m

            pred_pos = cde_model(cde_src.float()) #input acc_x, acc_y #(2048,64,1,100)

            pos_loss_cdist = torch.cdist(pred_pos, pose_xy[:, -1, 0:2].unsqueeze(1).float(), p=2).mean()
            pos_loss_l2 = torch.nn.functional.mse_loss(pred_pos, pose_xy[:, -1, 0:2].unsqueeze(1).float())
            pos_loss_l1 = torch.nn.functional.l1_loss(pred_pos, pose_xy[:, -1, 0:2].unsqueeze(1).float())
            fn_loss_cosim = loss_cosim()
            cur_cosim_loss = fn_loss_cosim(pred_pos, pose_xy[:, -1, 0:2].unsqueeze(1).float())
            cur_kl_loss = torch.abs(torch.nn.functional.kl_div(pred_pos, pose_xy[:, -1, 0:2].float()))
            loss = pos_loss_l2 + (pos_loss_l1) + (cur_cosim_loss) + cur_kl_loss + pos_loss_cdist

            #loss = pos_loss_l2 + pos_loss_cdist + pos_loss_l1

            running_l2_loss = running_l2_loss + pos_loss_l2
            running_cdist_loss = running_cdist_loss + pos_loss_cdist
            running_l1_loss = running_l1_loss + pos_loss_l1
            running_cosim_loss = running_cosim_loss + cur_cosim_loss
            running_kl_loss = running_kl_loss + cur_kl_loss
            running_loss = running_loss + loss

            print("Loss: ", '%.8f' % loss.item(), "cdist: ", '%.8f' % pos_loss_cdist.item(), "L2: ", '%.8f' % pos_loss_l2.item(), "L1: ", '%.8f' % pos_loss_l1.item(), "cosim: ", '%.8f' % cur_cosim_loss.item(), "kl: ", '%.8f' % cur_kl_loss.item())

            optimizer_AdamW_cde.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cde_model.parameters(), max_norm)
            optimizer_AdamW_cde.step()

        lr_scheduler_cde.step(loss)
        debug = True
        # if(epoch % 10 == 0) and (epoch > 0):
        if (debug == True):
            val_loss = []
            print("___________VALIDATION___________VALIDATION___________")
            with torch.no_grad():
                for i_val, data_val in enumerate(tqdm(validation_loader)):
                    data_m9, data_vicon = data_val
                    t_imu = data_m9[:, :, 0].unsqueeze(2).cuda()
                    acc_data = data_m9[:, :, 2:4].cuda()
                    gyro_z_imu = data_m9[:, :, 4].unsqueeze(2).cuda()
                    ofs = data_m9[:, :, 5:7].cuda()

                    pose_xy = data_vicon[:, :, 1:3].cuda() * scale # m

                    src_len = data_m9.shape[1]

                    cde_src = torch.cat([t_imu, acc_data, gyro_z_imu, ofs], dim=2).cuda()

                    pred_pos = cde_model(cde_src)  # input acc_x, acc_y #(2048,64,1,100)

                    pos_loss = torch.nn.functional.mse_loss(pred_pos, pose_xy[:, src_len - 1, 0:2].unsqueeze(1).float())
                    val_loss.append(pos_loss)
                    print("VALIDATION: Position_xy", '%.8f' % pos_loss.item())

                my_sum = 0
                for v in range(len(val_loss)):
                    my_sum = my_sum + val_loss[v]
                plot_val_loss = my_sum / len(val_loss)

                print("VALIDATION AVERAGE: Position_xy", '%.8f' % plot_val_loss.item())
                vis.line(Y=[plot_val_loss.detach().cpu().numpy()], X=np.array([epoch]), win=Validation_plot, update='append')

        torch.save({
            'epoch': epoch + 1,
            'loss': loss,

            'cde_model': cde_model.state_dict(),
            'optimizer_AdamW_cde': optimizer_AdamW_cde.state_dict(),
            'lr_scheduler_cde': lr_scheduler_cde.state_dict(),

        }, "/home/jsk/IMUlocalization/ckpt/save_ckpt.pth")


        vis.line(Y=[running_loss.detach().cpu().numpy()], X=np.array([epoch]), win=AllLoss, update='append')
        vis.line(Y=[running_cdist_loss.detach().cpu().numpy()], X=np.array([epoch]), win=plotcidst, update='append')
        vis.line(Y=[running_l2_loss.detach().cpu().numpy()], X=np.array([epoch]), win=plotL2, update='append')
        vis.line(Y=[running_l1_loss.detach().cpu().numpy()], X=np.array([epoch]), win=plotL1, update='append')
        vis.line(Y=[running_cosim_loss.detach().cpu().numpy()], X=np.array([epoch]), win=plotCosim, update='append')
        vis.line(Y=[running_kl_loss.detach().cpu().numpy()], X=np.array([epoch]), win=plotKL, update='append')

        print('epoch [{}/{}], loss:{:.4f} '.format(epoch + 1, num_epochs, running_loss))

if __name__ == '__main__':
    torch.cuda.empty_cache()

    solver = dict(method='rk4', rtol=1e-5, atol=1e-7)
    M9_train_pose_CDE_P(**solver)  ##210819