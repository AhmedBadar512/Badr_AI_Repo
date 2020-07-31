"""
    Model store which provides pretrained models.
"""

__all__ = ['get_model_file']

import os
import zipfile
import logging
import hashlib

_model_sha1 = {name: (error, checksum, repo_release_tag, ds, scale) for
               name, error, checksum, repo_release_tag, ds, scale in [
    ('alexnet', '1789', 'ecc4bb4e46e05dde17809978d2900f4fe14ea590', 'v0.0.422', 'in1k', 0.875),
    ('alexnetb', '1859', '9e390537e070ee42c5deeb6c456f81c991efbb49', 'v0.0.422', 'in1k', 0.875),
    ('zfnet', '1717', '9500db3008e9ca8bc8f8de8101ec760e5ac8c05a', 'v0.0.422', 'in1k', 0.875),
    ('zfnetb', '1480', '47533f6a367312c8b2f56202aeae0be366013116', 'v0.0.422', 'in1k', 0.875),
    ('vgg11', '1017', 'c20556f4179e9311f28baa310702b6ea9265fee8', 'v0.0.422', 'in1k', 0.875),
    ('vgg13', '0951', '9fa609fcb5cb44caf2737d13c0accc07cdea0c9d', 'v0.0.422', 'in1k', 0.875),
    ('vgg16', '0834', 'ce78831f5d0640bd2fd619ba7d8d5027e62eb4f2', 'v0.0.422', 'in1k', 0.875),
    ('vgg19', '0768', 'ec5ac0baa5d49c041af48e67d34d1a89f1a72e7f', 'v0.0.422', 'in1k', 0.875),
    ('bn_vgg11', '0936', 'ef31b86687e83d413cb9c95c9ead657c3de9f21b', 'v0.0.422', 'in1k', 0.875),
    ('bn_vgg13', '0887', '2cccc7252ab4798fd9a6c3ce9d0b59717c47e40b', 'v0.0.422', 'in1k', 0.875),
    ('bn_vgg16', '0759', '1ca9dee8ef41ed84a216636d3c21380988ea1bf8', 'v0.0.422', 'in1k', 0.875),
    ('bn_vgg19', '0688', '81d25be84932c1c2848cabd4533423e3fd2cdbec', 'v0.0.422', 'in1k', 0.875),
    ('bn_vgg11b', '0975', 'aeaccfdc4a655d895e280165cf5be856472ca91f', 'v0.0.422', 'in1k', 0.875),
    ('bn_vgg13b', '1019', '1102ffb7817ff11a8db85f1b9b8519b100da26a0', 'v0.0.422', 'in1k', 0.875),
    ('bn_vgg16b', '0862', '137178f78ace3943333a98d980dd88b4746e66af', 'v0.0.422', 'in1k', 0.875),
    ('bn_vgg19b', '0817', 'cd68a741183cbbab52562c4b7330d721e8ffa739', 'v0.0.422', 'in1k', 0.875),
    ('bninception', '0865', '4cab3cce0eb1b79b872b189f5b0d9e4bb20f5ff4', 'v0.0.423', 'in1k', 0.875),
    ('resnet10', '1390', '9e787f637312e04d3ec85136bf0ceca50acf8c80', 'v0.0.422', 'in1k', 0.875),
    ('resnet12', '1301', '8bc41d1b1da87463857bb5ca03fe252ef03116ad', 'v0.0.422', 'in1k', 0.875),
    ('resnet14', '1224', '7573d98872e622ef74e036c8a436a39ab75e9378', 'v0.0.422', 'in1k', 0.875),
    ('resnetbc14b', '1115', '5f30b7985b5a57d34909f3db08c52dfe1da065ac', 'v0.0.422', 'in1k', 0.875),
    ('resnet16', '1088', '14ce0d64680c3fe52f43b407a00d1a23b6cfd81c', 'v0.0.422', 'in1k', 0.875),
    ('resnet18_wd4', '1745', '6e80041645de7ccbe156ce5bc3cbde909cee6b41', 'v0.0.422', 'in1k', 0.875),
    ('resnet18_wd2', '1283', '85a7caff1b2f8e355a1b8cb559e836d5b0c22d12', 'v0.0.422', 'in1k', 0.875),
    ('resnet18_w3d4', '1067', 'c1735b7de29016779c95e8e1481e5ded955b2b63', 'v0.0.422', 'in1k', 0.875),
    ('resnet18', '0956', '6645845a7614afd265e997223d38e00433f00182', 'v0.0.422', 'in1k', 0.875),
    ('resnet26', '0837', 'a8f20f7194cdfcb6fd514a8dc9546105fd7a562a', 'v0.0.422', 'in1k', 0.875),
    ('resnetbc26b', '0757', 'd70a2cadfb648f4c528704f1b9983f35af94de6f', 'v0.0.422', 'in1k', 0.875),
    ('resnet34', '0744', '7f7d70e7780e24b4cb60cefc895198cdb2b94665', 'v0.0.422', 'in1k', 0.875),
    ('resnetbc38b', '0677', '75e405a71f7227de5abb6a3c3c44d807b5963c44', 'v0.0.422', 'in1k', 0.875),
    ('resnet50', '0604', '728800bf57bd49f79671399fd4fd2b7fe9883f07', 'v0.0.422', 'in1k', 0.875),
    ('resnet50b', '0614', 'b2a49da61dce6309c75e77226bb047b43247da24', 'v0.0.422', 'in1k', 0.875),
    ('resnet101', '0601', 'b6befeb4c8cf6d72d9c325c22df72ac792b51706', 'v0.0.422', 'in1k', 0.875),
    ('resnet101b', '0511', 'e3076227a06b394aebcce6260c4afc665224c987', 'v0.0.422', 'in1k', 0.875),
    ('resnet152', '0534', '2d8e394abcb9d35d2a853bb4dacb58460ff13551', 'v0.0.422', 'in1k', 0.875),
    ('resnet152b', '0480', 'b77f1e2c9158cc49deba2cf60b8a8e8d6605d654', 'v0.0.422', 'in1k', 0.875),
    ('preresnet10', '1402', '541bf0e17a576b1676069563a1ed0de0fde4090f', 'v0.0.422', 'in1k', 0.875),
    ('preresnet12', '1320', '349c0df4a835699bdb045bedc3d38a7747cd21d4', 'v0.0.422', 'in1k', 0.875),
    ('preresnet14', '1224', '194b876203e467fbad2ccd2e03b90a79bfec8dac', 'v0.0.422', 'in1k', 0.875),
    ('preresnetbc14b', '1152', 'bc4e06ff3df99e7ffa0b2bdafa224796fa46f5a9', 'v0.0.422', 'in1k', 0.875),
    ('preresnet16', '1080', 'e00c40ee6d211f553bff0274771e5461150c69f4', 'v0.0.422', 'in1k', 0.875),
    ('preresnet18_wd4', '1780', '6ac7bc592983ced18c863f203db80bbd30e87a0b', 'v0.0.422', 'in1k', 0.875),
    ('preresnet18_wd2', '1314', '0c0528c8ae4943aa68ba0298209f2ed418e4f644', 'v0.0.422', 'in1k', 0.875),
    ('preresnet18_w3d4', '1070', '056b46c6e8ee2c86ebee560efea81dd43bbd5de6', 'v0.0.422', 'in1k', 0.875),
    ('preresnet18', '0955', '621ead9297b93673ec1c040e091efff9142313b5', 'v0.0.422', 'in1k', 0.875),
    ('preresnet26', '0837', '1a92a73217b1611c27b0c7082a018328264a65ff', 'v0.0.422', 'in1k', 0.875),
    ('preresnetbc26b', '0788', '1f737cd6c173ed8e5d9a8a69b35e1cf696ba622e', 'v0.0.422', 'in1k', 0.875),
    ('preresnet34', '0754', '3cc5ae1481512a8b206fb96ac8b632bcc5ee2db9', 'v0.0.422', 'in1k', 0.875),
    ('preresnetbc38b', '0636', '3396b49b5d20e7d362f9bd8879c00a21e8d67df1', 'v0.0.422', 'in1k', 0.875),
    ('preresnet50', '0625', '208605629d347a64b9a354f5ad7f441f736eb418', 'v0.0.422', 'in1k', 0.875),
    ('preresnet50b', '0634', '711227b1a93dd721dd3e37709456acfde969ba18', 'v0.0.422', 'in1k', 0.875),
    ('preresnet101', '0573', 'd45ea488f72fb99af1c46e4064b12c5014a7b626', 'v0.0.422', 'in1k', 0.875),
    ('preresnet101b', '0539', '54d23aff956752be614c2ba66d8bff5477cf0367', 'v0.0.422', 'in1k', 0.875),
    ('preresnet152', '0532', '0ad4b58f2365028db9216f1e080898284328cc3e', 'v0.0.422', 'in1k', 0.875),
    ('preresnet152b', '0500', '119062d97d30f6636905c824c6d1b4e21be2c3f2', 'v0.0.422', 'in1k', 0.875),
    ('preresnet200b', '0563', '2f9c761d78714c33d3b260add782e3851b0078f4', 'v0.0.422', 'in1k', 0.875),
    ('preresnet269b', '0557', '7003b3c4a1dea496f915750b4411cc67042a111d', 'v0.0.422', 'in1k', 0.875),
    ('resnext14_16x4d', '1222', 'bff90c1d3dbde7ea4a6972bbacb619e252d344ea', 'v0.0.422', 'in1k', 0.875),
    ('resnext14_32x2d', '1247', '06aa6709cfb4cf23793eb0eee5d5fce42cfcb9cb', 'v0.0.422', 'in1k', 0.875),
    ('resnext14_32x4d', '1115', '3acdaec14a6c74284c03bc79ed47e9ecb394e652', 'v0.0.422', 'in1k', 0.875),
    ('resnext26_32x2d', '0851', '827791ccefaef07e5837f8fb1dae8733c871c029', 'v0.0.422', 'in1k', 0.875),
    ('resnext26_32x4d', '0718', '4f05525e34b9aeb82db2339f714b25055d94657b', 'v0.0.422', 'in1k', 0.875),
    ('resnext50_32x4d', '0547', '45234d14f0e80700afc5c61e1bd148d848d8d089', 'v0.0.422', 'in1k', 0.875),
    ('resnext101_32x4d', '0494', '3990ddd1e776c7e90625db9a8f683e1d6a6fb301', 'v0.0.422', 'in1k', 0.875),
    ('resnext101_64x4d', '0484', 'f8cf1580943cf3c6d6019f2fcc44f8adb857cb20', 'v0.0.422', 'in1k', 0.875),
    ('seresnet10', '1332', '33a592e1497d37a427920c1408be908ba28d2a6d', 'v0.0.422', 'in1k', 0.875),
    ('seresnet18', '0921', '46c847abfdbd82c41a096e385163f21ae29ee200', 'v0.0.422', 'in1k', 0.875),
    ('seresnet26', '0807', '5178b3b1ea71bb118ffcc5d471f782f4ae6150d4', 'v0.0.422', 'in1k', 0.875),
    ('seresnetbc26b', '0684', '1460a381603c880f24fb0a42bfb6b79b850e2b28', 'v0.0.422', 'in1k', 0.875),
    ('seresnetbc38b', '0575', '18fcfcc1fee078382ad957e0f7d139ff596732e7', 'v0.0.422', 'in1k', 0.875),
    ('seresnet50', '0560', 'f1b84c8de0d25bbd4e92fcaefd9dd5012fa74bc4', 'v0.0.441', 'in1k', 0.875),
    ('seresnet50b', '0533', '256002c3b489d5b685ee1ab6b62303d7768c5816', 'v0.0.422', 'in1k', 0.875),
    ('seresnet101', '0589', '2a22ba87f5b0d56d51063898161d4c42cac45325', 'v0.0.422', 'in1k', 0.875),
    ('seresnet101b', '0464', 'a10be1d25d3112825e7b77277d6c56eb276dc799', 'v0.0.460', 'in1k', 0.875),
    ('seresnet152', '0576', '8023259a13a53aa0a72d9df6468721314e702872', 'v0.0.422', 'in1k', 0.875),
    ('sepreresnet10', '1309', 'af20d06c486dc97cff0f6d9bc52a7c7458040514', 'v0.0.422', 'in1k', 0.875),
    ('sepreresnet18', '0940', 'fe403280f68a5dfa93366437b9ff37ce3a419cf8', 'v0.0.422', 'in1k', 0.875),
    ('sepreresnetbc26b', '0640', 'a72bf8765efb1024bdd33eebe9920fd3e22d0bd6', 'v0.0.422', 'in1k', 0.875),
    ('sepreresnetbc38b', '0567', '17d10c63f096db1b7bfb59b6c6ffe14b9c669676', 'v0.0.422', 'in1k', 0.875),
    ('sepreresnet50b', '0531', '0882c0e9add4dad0304443fa8a704ee28c5e1c58', 'v0.0.461', 'in1k', 0.875),
    ('seresnext50_32x4d', '0509', '4244900a583098a5fb6c174c834f44a7471305c2', 'v0.0.422', 'in1k', 0.875),
    ('seresnext101_32x4d', '0459', '13a9b2fd699a3e25ee18d93a408dbaf3dee74428', 'v0.0.422', 'in1k', 0.875),
    ('seresnext101_64x4d', '0465', 'ec0a3b132256c8a7d0f92c45775d201a456f25fb', 'v0.0.422', 'in1k', 0.875),
    ('senet16', '0805', 'f5f576568d02a572be5276b0b64e71ce4d1c4531', 'v0.0.422', 'in1k', 0.875),
    ('senet28', '0590', '667d56873564cc22b2f10478d5f3d55cda580c61', 'v0.0.422', 'in1k', 0.875),
    ('senet154', '0466', 'f1b79a9bf0f7073bacf534d846c03d1b71dc404b', 'v0.0.422', 'in1k', 0.875),
    ('ibn_resnet50', '0668', '4c72a071e13235ccea0db3d932db8ec5f691e155', 'v0.0.427', 'in1k', 0.875),
    ('ibn_resnet101', '0584', '2c2c4993de8b8d79a66a62a1dbf682e552eb16c1', 'v0.0.427', 'in1k', 0.875),
    ('ibnb_resnet50', '0695', '7178cc50d166fa2d2474b5110aaea7fcd41bd8ca', 'v0.0.427', 'in1k', 0.875),
    ('ibn_resnext101_32x4d', '0564', 'c149beb5a735b75d35a728f0f0054514899e9f8b', 'v0.0.427', 'in1k', 0.875),
    ('ibn_densenet121', '0749', '009d1919ec097777b9ffb3c1c4ff7802e0158201', 'v0.0.427', 'in1k', 0.875),
    ('ibn_densenet169', '0684', '7152d6ccf07babca362df603d45b09fd37ca6744', 'v0.0.427', 'in1k', 0.875),
    ('airnet50_1x64d_r2', '0623', '6940f0e553a65c1beb4b769e31685cdde59359b8', 'v0.0.423', 'in1k', 0.875),
    ('airnet50_1x64d_r16', '0650', 'b7bb86623e680f08a39828894052099cc5198842', 'v0.0.423', 'in1k', 0.875),
    ('airnext50_32x4d_r2', '0572', 'fa8e40ab400cd8507a02606db72d270382482ecf', 'v0.0.423', 'in1k', 0.875),
    ('bam_resnet50', '0697', '3a4101c80ee21a615835f954c5ca67a959978554', 'v0.0.424', 'in1k', 0.875),
    ('cbam_resnet50', '0639', '1d0bdb0e36545428975df6dcb32bac876934744c', 'v0.0.429', 'in1k', 0.875),
    ('pyramidnet101_a360', '0651', '9db84918734d8fe916664ecef49df0a0c0168530', 'v0.0.423', 'in1k', 0.875),
    ('diracnet18v2', '1113', '4d687b749342d23996d078a0984fd6affe63e47c', 'v0.0.429', 'in1k', 0.875),
    ('diracnet34v2', '0950', '161d97fda4104be091e918ea24c903bfffdc9b8d', 'v0.0.429', 'in1k', 0.875),
    ('densenet121', '0684', 'e9196a9c93534ca7b71ef136e5cc27f240370481', 'v0.0.422', 'in1k', 0.875),
    ('densenet161', '0591', '78224027b390f943b30130a7921ded2887776a77', 'v0.0.432', 'in1k', 0.875),
    ('densenet169', '0606', 'f708dc3310008e59814745ffc22ddf829fb2d25a', 'v0.0.422', 'in1k', 0.875),
    ('densenet201', '0591', '450c656858d693932253b486069690fe727f6f89', 'v0.0.426', 'in1k', 0.875),
    ('peleenet', '1129', 'e1c3cdea31e2c683d71f808765963c2fffcd672e', 'v0.0.429', 'in1k', 0.875),
    ('wrn50_2', '0614', 'bea17aa953afed82540c509d7c2964d602fcb2af', 'v0.0.423', 'in1k', 0.875),
    ('drnc26', '0788', '571eb2dc632b9aecd2726507847412e4e2d3149b', 'v0.0.425', 'in1k', 0.875),
    ('drnc42', '0693', '52dd60289e5d9cd8eeb66786eb31b9bd5b1b0b36', 'v0.0.425', 'in1k', 0.875),
    ('drnc58', '0626', 'e5c7be8922e6c9e60661d0aa88618f5b28961289', 'v0.0.425', 'in1k', 0.875),
    ('drnd22', '0848', '42f7a37bc912979db496fff8b808f724b4712974', 'v0.0.425', 'in1k', 0.875),
    ('drnd38', '0737', 'a110827559aa831a3b2b9a2b032c8adbc47769e5', 'v0.0.425', 'in1k', 0.875),
    ('drnd54', '0626', 'cb792485021c6f946e28cc3e72674e5a1286b9da', 'v0.0.425', 'in1k', 0.875),
    ('drnd105', '0583', '80eb9ec2efd053d2f1e73d08911208c5d787e7cf', 'v0.0.425', 'in1k', 0.875),
    ('dpn68', '0658', '5b70b7b86c33c3dfb04f5fa189e5d501e8804499', 'v0.0.427', 'in1k', 0.875),
    ('dpn98', '0528', '6883ec37bc83f092101511a4e46702f1587f970e', 'v0.0.427', 'in1k', 0.875),
    ('dpn131', '0524', '971af47c5c45175a9999002849d4bb5e47fa99f3', 'v0.0.427', 'in1k', 0.875),
    ('darknet_tiny', '1745', 'd30be41aad15edf40dfed0bbf53d0e68c520f9f3', 'v0.0.422', 'in1k', 0.875),
    ('darknet_ref', '1671', 'b4991f6b58ae95118aa9ea84cae4a27e328196b5', 'v0.0.422', 'in1k', 0.875),
    ('darknet53', '0558', '4a63ab3005e5138445da5fac4247c460de02a41b', 'v0.0.422', 'in1k', 0.875),
    ('bagnet9', '3553', '43eb57dcbbce90287d0c3158457077fcc6a4c5ef', 'v0.0.424', 'in1k', 0.875),
    ('bagnet17', '2154', '8a31e34793f4ebc9c7585f531dab1b47b3befc0d', 'v0.0.424', 'in1k', 0.875),
    ('bagnet33', '1497', 'ef600c89aacdd881c2c5483defa9cb220286d31b', 'v0.0.424', 'in1k', 0.875),
    ('dla34', '0823', '9232e3e7c299c2e83a49e5372affee2f19226518', 'v0.0.427', 'in1k', 0.875),
    ('dla46c', '1287', 'dfcae3b549121205008235fd7e59793b394f8998', 'v0.0.427', 'in1k', 0.875),
    ('dla46xc', '1229', 'a858beca359f41cfe836cec6d30b01ba98109d06', 'v0.0.427', 'in1k', 0.875),
    ('dla60', '0711', '7375fcfd8ec94bfd6587ef49d52e4f2dcefc0296', 'v0.0.427', 'in1k', 0.875),
    ('dla60x', '0621', '3c5941dbfdf66b879c02901282e8500288bc6498', 'v0.0.427', 'in1k', 0.875),
    ('dla60xc', '1075', 'a7850f0307de77fcce42afdbb7070776b7c219ca', 'v0.0.427', 'in1k', 0.875),
    ('dla102', '0643', '2be886b250ba9ea8721e8bdba62b4e32d33e19e4', 'v0.0.427', 'in1k', 0.875),
    ('dla102x', '0602', '46640eec0179abf109951d865f5c397024cf9297', 'v0.0.427', 'in1k', 0.875),
    ('dla102x2', '0553', '06c930313e017f2ef9596d9259f0029d399f563a', 'v0.0.427', 'in1k', 0.875),
    ('dla169', '0590', 'e010166d75cd6603a94f006f0dbf5a4d9185bf07', 'v0.0.427', 'in1k', 0.875),
    ('hrnet_w18_small_v1', '0974', '8db99936134de71bf0700e2855b5caef30a95298', 'v0.0.428', 'in1k', 0.875),
    ('hrnet_w18_small_v2', '0805', 'fcb8e21898d1dd5ace4587f33e1d5e9c335369e5', 'v0.0.428', 'in1k', 0.875),
    ('hrnetv2_w18', '0686', '71c614d725ecfca2506ccf5d71723796cc7ae275', 'v0.0.428', 'in1k', 0.875),
    ('hrnetv2_w30', '0606', '4883e3451691d7d14a3d7d3572aecc21f3aa8454', 'v0.0.428', 'in1k', 0.875),
    ('hrnetv2_w32', '0607', 'ef949840f95a1cd82bc7ad8795929c795058d78b', 'v0.0.428', 'in1k', 0.875),
    ('hrnetv2_w40', '0573', '29cece1c277ee70a91a373f3c5cb266f6a1af9e3', 'v0.0.428', 'in1k', 0.875),
    ('hrnetv2_w44', '0595', 'a4e4781ca1c32fc98beed3167832601ca51266c9', 'v0.0.428', 'in1k', 0.875),
    ('hrnetv2_w48', '0581', '3af4ed57e2c7dab91794f933f7e8105320935d31', 'v0.0.428', 'in1k', 0.875),
    ('hrnetv2_w64', '0553', 'aede8def2f12173f640f85187b531c5218615d92', 'v0.0.428', 'in1k', 0.875),
    ('vovnet39', '0825', '49cbcdc62cc7815a4bc76e508da8feee0f802e00', 'v0.0.431', 'in1k', 0.875),
    ('vovnet57', '0812', '0977958aceb28a7481a230e0ba52750d43e5b152', 'v0.0.431', 'in1k', 0.875),
    ('selecsls42b', '0676', '0d785bec0c31aee57e1d267900ae1a942a665fcb', 'v0.0.430', 'in1k', 0.875),
    ('selecsls60', '0630', 'a799a0e5ddcc3991808bd8d98a83a3e717ee87e4', 'v0.0.430', 'in1k', 0.875),
    ('selecsls60b', '0604', 'bc9c43191043382b38e3be5893d1d8316ca401e9', 'v0.0.430', 'in1k', 0.875),
    ('hardnet39ds', '1003', '4971cd5a76946293a137d78032ee024f0258c979', 'v0.0.435', 'in1k', 0.875),
    ('hardnet68ds', '0845', 'dd35f3f91bfe55c354d4aac2b5830c3a744741ed', 'v0.0.435', 'in1k', 0.875),
    ('hardnet68', '0740', '9ea05e3973dddb52b970872fc3ed76fa32d10731', 'v0.0.435', 'in1k', 0.875),
    ('hardnet85', '0644', '7892e2215c2d1c32996be09a724c8125c8c49572', 'v0.0.435', 'in1k', 0.875),
    ('squeezenet_v1_0', '1760', 'd13ba73265325f21eb34e782989a7269cad406c6', 'v0.0.422', 'in1k', 0.875),
    ('squeezenet_v1_1', '1742', '95b614487f1f0572bd0dba18e0fc6d63df3a6bfc', 'v0.0.422', 'in1k', 0.875),
    ('squeezeresnet_v1_0', '1783', 'db620d998257c84fd6d5e80bba48cc1022febda3', 'v0.0.422', 'in1k', 0.875),
    ('squeezeresnet_v1_1', '1789', '13d6bc6bd85adf83ef55325443495feb07c5788f', 'v0.0.422', 'in1k', 0.875),
    ('sqnxt23_w1', '1861', '379975ebe54b180f52349c3737b17ea7b2613953', 'v0.0.422', 'in1k', 0.875),
    ('sqnxt23v5_w1', '1762', '153b4ce73714d2ecdca294efb365ab9c026e2f41', 'v0.0.422', 'in1k', 0.875),
    ('sqnxt23_w3d2', '1334', 'a2ba956cfeed0b4bbfc37776c6a1cd5ca13d9345', 'v0.0.422', 'in1k', 0.875),
    ('sqnxt23v5_w3d2', '1284', '72efaa710f0f1645cb220cb9950b3660299f2bed', 'v0.0.422', 'in1k', 0.875),
    ('sqnxt23_w2', '1069', 'f43dee19c527460f9815fc4e5eeeaef99fae4df3', 'v0.0.422', 'in1k', 0.875),
    ('sqnxt23v5_w2', '1026', 'da80c6407a4c18be31bcdd08356666942a9ef2b4', 'v0.0.422', 'in1k', 0.875),
    ('shufflenet_g1_wd4', '3681', '04a9e2d4ada22b3d317e2fc8b7d4ec11865c414f', 'v0.0.422', 'in1k', 0.875),
    ('shufflenet_g3_wd4', '3618', 'c9aad0f08d129726bbc19219c9773b38cf38825e', 'v0.0.422', 'in1k', 0.875),
    ('shufflenet_g1_wd2', '2236', '082db702c422d8bce12d4d79228de56f088a420d', 'v0.0.422', 'in1k', 0.875),
    ('shufflenet_g3_wd2', '2059', 'e3aefeeb36c20e325d0c7fe46afc60484167609d', 'v0.0.422', 'in1k', 0.875),
    ('shufflenet_g1_w3d4', '1679', 'a1cc5da3a288299a33353f697ed0297328dc3e95', 'v0.0.422', 'in1k', 0.875),
    ('shufflenet_g3_w3d4', '1611', '89546a05f499f0fdf96dade0f3db430f92c5920d', 'v0.0.422', 'in1k', 0.875),
    ('shufflenet_g1_w1', '1348', '52ddb20fd7ff288ae30a17757efda4653c09d5ca', 'v0.0.422', 'in1k', 0.875),
    ('shufflenet_g2_w1', '1333', '2a8ba6928e6fac05a5fe8911a9a175268eb18382', 'v0.0.422', 'in1k', 0.875),
    ('shufflenet_g3_w1', '1326', 'daaec8b84572023c1352e11830d296724123408e', 'v0.0.422', 'in1k', 0.875),
    ('shufflenet_g4_w1', '1313', '35dbd6b9fb8bc3e97367ea210abbd61da407f226', 'v0.0.422', 'in1k', 0.875),
    ('shufflenet_g8_w1', '1322', '449fb27659101a2cf0a87c90e33f4632d1c5e9f2', 'v0.0.422', 'in1k', 0.875),
    ('shufflenetv2_wd2', '1843', 'd492d721d3167cd64ab1c2a1f33f3ca5f6dec7c3', 'v0.0.422', 'in1k', 0.875),
    ('shufflenetv2_w1', '1135', 'dae13ee9f24c89cd1ea12a58fb90b967223c8e2e', 'v0.0.422', 'in1k', 0.875),
    ('shufflenetv2_w3d2', '0923', 'ea615baab737fca3a3d90303844b4a2922ea2c62', 'v0.0.422', 'in1k', 0.875),
    ('shufflenetv2_w2', '0821', '6ccac868f595e4618ca7e5f67f7c113f021ffad4', 'v0.0.422', 'in1k', 0.875),
    ('shufflenetv2b_wd2', '1784', 'd5644a6ab8fcb6ff04f30a2eb862ebd2de92b94c', 'v0.0.422', 'in1k', 0.875),
    ('shufflenetv2b_w1', '1104', 'b7db0ca041e996ee76fec7f126dc39c4e5120e82', 'v0.0.422', 'in1k', 0.875),
    ('shufflenetv2b_w3d2', '0877', '9efb13f7d795d63c8fbee736622b9f1940dd5dd5', 'v0.0.422', 'in1k', 0.875),
    ('shufflenetv2b_w2', '0808', 'ba5c7ddcd8f7da3719f5d1de71d5fd30130d59d9', 'v0.0.422', 'in1k', 0.875),
    ('menet108_8x1_g3', '2039', '1a8cfc9296011cd994eb48e75e24c33ecf6580f5', 'v0.0.422', 'in1k', 0.875),
    ('menet128_8x1_g4', '1918', '7fb59f0a8d3e1f490c26546dfe93ea29ebd79c2b', 'v0.0.422', 'in1k', 0.875),
    ('menet160_8x1_g8', '2034', '3cf9eb2aa2d4e067aa49ce32e7a41e9db5262493', 'v0.0.422', 'in1k', 0.875),
    ('menet228_12x1_g3', '1291', '21bd19bf0adb73b10cb04ccce8688f119467a114', 'v0.0.422', 'in1k', 0.875),
    ('menet256_12x1_g4', '1217', 'd9f2e10e6402e5ee2aec485da07da72edf25f790', 'v0.0.422', 'in1k', 0.875),
    ('menet348_12x1_g3', '0937', 'cee7691c710f5c453b63ef9e8c3e15e699b004bb', 'v0.0.422', 'in1k', 0.875),
    ('menet352_12x1_g8', '1167', '54a916bcc3920c6ef24243c8c73604b25d728a6d', 'v0.0.422', 'in1k', 0.875),
    ('menet456_24x1_g3', '0779', '2a70b14bd17e8d4692f15f2f8e9d181e7d95b971', 'v0.0.422', 'in1k', 0.875),
    ('mobilenet_wd4', '2213', 'ad04596aa730e5bb4429115df70504c5a7dd5969', 'v0.0.422', 'in1k', 0.875),
    ('mobilenet_wd2', '1333', '01395e1b9e2a54065aafcc8b4c419644e7f6a655', 'v0.0.422', 'in1k', 0.875),
    ('mobilenet_w3d4', '1051', '7832561b956f0d763b002fbd9f2f880bbb712885', 'v0.0.422', 'in1k', 0.875),
    ('mobilenet_w1', '0866', '6939232b46fb98c8a9209d66368d630bb50941ed', 'v0.0.422', 'in1k', 0.875),
    ('fdmobilenet_wd4', '3062', '36aa16df43b344f42d6318cc840a81702951a033', 'v0.0.422', 'in1k', 0.875),
    ('fdmobilenet_wd2', '1977', '34541b84660b4e812830620c5d48df7c7a142078', 'v0.0.422', 'in1k', 0.875),
    ('fdmobilenet_w3d4', '1597', '0123c0313194a3094ec006f757d93f59aad73c2b', 'v0.0.422', 'in1k', 0.875),
    ('fdmobilenet_w1', '1312', 'fa99fb8d728f66f68464221e049a33cd2b8bfc6a', 'v0.0.422', 'in1k', 0.875),
    ('mobilenetv2_wd4', '2413', 'c3705f55b0df68919fba7ed79204c5651f6f71b1', 'v0.0.422', 'in1k', 0.875),
    ('mobilenetv2_wd2', '1446', 'b0c9a98b85b579ba77c17d228ace399809c6ab43', 'v0.0.422', 'in1k', 0.875),
    ('mobilenetv2_w3d4', '1044', 'e122c73eae885d204bc2ba46fb013a9da5cb282f', 'v0.0.422', 'in1k', 0.875),
    ('mobilenetv2_w1', '0863', 'b32cede3b68f40f2ed0552dcdf238c70f82e5705', 'v0.0.422', 'in1k', 0.875),
    ('mobilenetv2b_wd4', '2505', '4079f654160fcd7ce37212c2d2da424f22913a70', 'v0.0.453', 'in1k', 0.875),
    ('mobilenetv2b_wd2', '1473', '129cfd918a4c4ab1b2544bc9ae24ed03f97f00ee', 'v0.0.453', 'in1k', 0.875),
    ('mobilenetv2b_w3d4', '1152', 'fa93741abc02bf3feb9e276d26dcfc6249a7b0a4', 'v0.0.453', 'in1k', 0.875),
    ('mobilenetv2b_w1', '0943', '97f1b6763b054d821e3422c7b4faf990005ec2c0', 'v0.0.453', 'in1k', 0.875),
    ('mobilenetv3_large_w1', '0769', 'f66596ae10c8eaa1ea3cb79b2645bd93f946b059', 'v0.0.422', 'in1k', 0.875),
    ('igcv3_wd4', '2828', '309359dc5a0cd0439f2be5f629534aa3bdf2b4f9', 'v0.0.422', 'in1k', 0.875),
    ('igcv3_wd2', '1701', 'b952333ab2024f879d4bb9895331a617f2b957b5', 'v0.0.422', 'in1k', 0.875),
    ('igcv3_w3d4', '1100', '00294c7b1ab9dddf7ab2cef3e7ec0a627bd67b29', 'v0.0.422', 'in1k', 0.875),
    ('igcv3_w1', '0899', 'a0cb775dd5bb2c13dce35a21d6fd53a783959702', 'v0.0.422', 'in1k', 0.875),
    ('mnasnet_b1', '0802', '763d6849142ce86f46cb7ec4c003ccf15542d6eb', 'v0.0.422', 'in1k', 0.875),
    ('mnasnet_a1', '0756', '8e0f49481a3473b9457d0987c9c6f7e51ff57576', 'v0.0.422', 'in1k', 0.875),
    ('proxylessnas_cpu', '0751', '47e1431680e115462835e73ec21dec8b6e88eb13', 'v0.0.424', 'in1k', 0.875),
    ('proxylessnas_gpu', '0726', 'd536cb3e27a47a4a18aa8e230ebe6b4a8f748910', 'v0.0.424', 'in1k', 0.875),
    ('proxylessnas_mobile', '0783', 'da8cdb80c5bd618258c657ebd8506e1342eaeb0d', 'v0.0.424', 'in1k', 0.875),
    ('proxylessnas_mobile14', '0653', '478b58cdb6c94007f786ec06a9e71a8dbc14507f', 'v0.0.424', 'in1k', 0.875),
    ('fbnet_cb', '0784', 'acd12097da630e4bf9051d138f04c7e9535e58c1', 'v0.0.428', 'in1k', 0.875),
    ('xception', '0558', 'b95b50510de4e39e2ddf759e69501a7470787c00', 'v0.0.423', 'in1k', 0.875),
    ('inceptionv3', '0563', 'b0094c1c279551394aa5c9709003c567324dcd70', 'v0.0.427', 'in1k', 0.875),
    ('inceptionv4', '0541', 'c1fa5642c0218e89fbe3effb233bffeb24672ba9', 'v0.0.428', 'in1k', 0.875),
    ('inceptionresnetv2', '0495', '3e2cc5456bb14fbdaec55006430278970ab64050', 'v0.0.428', 'in1k', 0.875),
    ('polynet', '0451', 'e752c86bbde4f5ce07ab6d079673a62a7565acf7', 'v0.0.428', 'in1k', 0.875),
    ('nasnet_4a1056', '0833', '9710e638693fa52538b268e767706210bf37d667', 'v0.0.428', 'in1k', 0.875),
    ('nasnet_6a4032', '0427', '1f0d2198bffb71386290b9b4e2058af2610574d8', 'v0.0.428', 'in1k', 0.875),
    ('pnasnet5large', '0427', '90e804af249c36f5f4435eb58ee0f32debefb320', 'v0.0.428', 'in1k', 0.875),
    ('spnasnet', '0873', 'a38a57a3d582ec4e227405924b84928587ea362f', 'v0.0.427', 'in1k', 0.875),
    ('efficientnet_b0', '0725', 'fc13925b2b95f5469aba2bb7b8472fdbabd663c3', 'v0.0.427', 'in1k', 0.875),
    ('efficientnet_b1', '0630', '82e0c512dc557ccb4eb3fbdabf48106988251d6d', 'v0.0.427', 'in1k', 0.882),
    ('efficientnet_b0b', '0668', '771272448df362b9637c7edf94292ab2c9676314', 'v0.0.429', 'in1k', 0.875),
    ('efficientnet_b1b', '0577', 'b294ee16111847f37129ff069f9911f76a2233d4', 'v0.0.429', 'in1k', 0.882),
    ('efficientnet_b2b', '0530', '55bcdc5d03493a581c3a3778b5ee6c08142718b4', 'v0.0.429', 'in1k', 0.890),
    ('efficientnet_b3b', '0469', 'b8210e1ac4f331b25b95c4a6d30e4b024d84ceb3', 'v0.0.429', 'in1k', 0.904),
    ('efficientnet_b4b', '0399', '5e35e9c56c3a0f705a44a38087e2084a25ee0a2e', 'v0.0.429', 'in1k', 0.922),
    ('efficientnet_b5b', '0343', '0ed0c69daa1d75e2da35f49ddea6bcfa0383727f', 'v0.0.429', 'in1k', 0.934),
    ('efficientnet_b6b', '0312', 'faf631041f84b19668eb207201ec13b2d405e702', 'v0.0.429', 'in1k', 0.942),
    ('efficientnet_b7b', '0315', '4024912ec1499b559de26b2ee7d7be1c2a3e53cf', 'v0.0.429', 'in1k', 0.949),
    ('efficientnet_b0c', '0646', '2bd0e2af1d275ab2046002719305bf517137f6df', 'v0.0.433', 'in1k', 0.875),
    ('efficientnet_b1c', '0582', 'a760b325d867a5aa4093ae69d68e8df04ed7730b', 'v0.0.433', 'in1k', 0.882),
    ('efficientnet_b2c', '0533', 'ea6ca9cf3c5179ad3927d7c3386c1c18c7183e24', 'v0.0.433', 'in1k', 0.890),
    ('efficientnet_b3c', '0464', '1c8fced86bc52d3d97fdce3750180d6b694f53c6', 'v0.0.433', 'in1k', 0.904),
    ('efficientnet_b4c', '0390', 'dc4379eac0dc4144260a270d4eb4ea3835394703', 'v0.0.433', 'in1k', 0.922),
    ('efficientnet_b5c', '0310', '80258ef75ea1b068b6ccf66420b8dd346c0bcdaa', 'v0.0.433', 'in1k', 0.934),
    ('efficientnet_b6c', '0286', '285f830add2ce100c6ab035f2a0caf49a33308ad', 'v0.0.433', 'in1k', 0.942),
    ('efficientnet_b7c', '0276', '1ffad4eca775d49ba48a0aa168a9c81649dab5b1', 'v0.0.433', 'in1k', 0.949),
    ('efficientnet_b8c', '0270', 'aa691b94070f49e2b7f3a0ac11bc5ddbdb18b1f6', 'v0.0.433', 'in1k', 0.954),
    ('efficientnet_edge_small_b', '0642', '1c03bb7355c6ab14374520743cc56e1ee22e773b', 'v0.0.434', 'in1k', 0.875),
    ('efficientnet_edge_medium_b', '0565', '73153b188d8b79cd8cc0ab45991561499df87838', 'v0.0.434', 'in1k', 0.882),
    ('efficientnet_edge_large_b', '0496', 'd72edce103b4bdac37afeabec281f1aedc9632bc', 'v0.0.434', 'in1k', 0.904),
    ('mixnet_s', '0737', 'd68d63f1914beeaec4e068c0dbd1defe09c7ffb6', 'v0.0.427', 'in1k', 0.875),
    ('mixnet_m', '0679', 'f74eab6c0ed1bc453453f433bce1cdde2d3e6bda', 'v0.0.427', 'in1k', 0.875),
    ('mixnet_l', '0601', '5c2ccc0c906ae29985043dc590317133c0be3376', 'v0.0.427', 'in1k', 0.875),
    ('resneta50b', '0543', 'ba99aca40e3c117e94980b0d6786910eeae7b9ee', 'v0.0.452', 'in1k', 0.875),
    ('resneta101b', '0490', 'd6dfa5240ecfa2c11f851535c9e2dafdc3bf016f', 'v0.0.452', 'in1k', 0.875),
    ('resneta152b', '0465', 'a54b896fcef292ad3e5d6d1290e83cb760d97084', 'v0.0.452', 'in1k', 0.875),
    ('resnetd50b', '0549', '1c84294f68b78dc58e07496495be0f8ecd2f14e3', 'v0.0.447', 'in1k', 0.875),
    ('resnetd101b', '0459', '7cce7f1357a3de297f7000b33f505dc67c38fb96', 'v0.0.447', 'in1k', 0.875),
    ('resnetd152b', '0468', '4673f64c71cf438eeafc890b5a138e301437bf90', 'v0.0.447', 'in1k', 0.875),
    ('resnet20_cifar10', '0597', '451230e98c5da3cd24e364b76995cdf5bdd36b73', 'v0.0.438', 'cf', 0.0),
    ('resnet20_cifar100', '2964', '5fa28f78b6b33f507f6b79a41f7fca07f681e4a5', 'v0.0.438', 'cf', 0.0),
    ('resnet20_svhn', '0343', '3480eec0f2781350815d07aa57bb821ecadc8b69', 'v0.0.438', 'cf', 0.0),
    ('resnet56_cifar10', '0452', 'a39ad94af7aad7adf21f41436cb8d86a948c7e90', 'v0.0.438', 'cf', 0.0),
    ('resnet56_cifar100', '2488', '8e413ab97ce41f96e02888776bc9ec71df49d909', 'v0.0.438', 'cf', 0.0),
    ('resnet56_svhn', '0275', '5acc55374dab36f2ebe70948393112fad83c4b17', 'v0.0.438', 'cf', 0.0),
    ('resnet110_cifar10', '0369', 'c625643a3c10909cdfc6c955418f0fca174b8d01', 'v0.0.438', 'cf', 0.0),
    ('resnet110_cifar100', '2280', 'c248211b354f7058b3066c5fb4ad87b2d0bdb6a0', 'v0.0.438', 'cf', 0.0),
    ('resnet110_svhn', '0245', 'a07e849f5e3233ef458072a30d8cc04ae84ff054', 'v0.0.438', 'cf', 0.0),
    ('resnet164bn_cifar10', '0368', 'cf08cca79ac123304add47b3aaba11cb4c46a25b', 'v0.0.438', 'cf', 0.0),
    ('resnet164bn_cifar100', '2044', '1ba347905fe05d922c9ec5ba876611b6393c6c99', 'v0.0.438', 'cf', 0.0),
    ('resnet164bn_svhn', '0242', '1bfa8083c38c89c19a4e0b53f714876705624fa7', 'v0.0.438', 'cf', 0.0),
    ('resnet272bn_cifar10', '0333', 'c8b0a926aeba2cdd404454bb22a731a3aed5996c', 'v0.0.438', 'cf', 0.0),
    ('resnet272bn_cifar100', '2007', '5357e0df7431ce2fb41f748fa04454f5a7055d1c', 'v0.0.438', 'cf', 0.0),
    ('resnet272bn_svhn', '0243', 'e2a8e35588d6375815a9b633f66e019a393553f7', 'v0.0.438', 'cf', 0.0),
    ('resnet542bn_cifar10', '0343', 'c31829d4c5845f9604e1a0f5aec938f03fcc05c3', 'v0.0.438', 'cf', 0.0),
    ('resnet542bn_cifar100', '1932', '2db913a6e6e577a366e2ab30030b9e976a388008', 'v0.0.438', 'cf', 0.0),
    ('resnet542bn_svhn', '0234', '0d6759e722dd536b2ce16ef856b6926fba023c6d', 'v0.0.438', 'cf', 0.0),
    ('resnet1001_cifar10', '0328', '552ab287f0a8224ae960a4ec0b4aed0f309e6641', 'v0.0.438', 'cf', 0.0),
    ('resnet1001_cifar100', '1979', '75c8acac55fce2dfc5c3f56cd10dd0467e56ffd2', 'v0.0.438', 'cf', 0.0),
    ('resnet1001_svhn', '0241', 'c9a01550d011abc9e6bc14df63952715a88a506a', 'v0.0.438', 'cf', 0.0),
    ('resnet1202_cifar10', '0353', '3559a9431d3ddd3ef1ee24bf2baa1b7184a21108', 'v0.0.438', 'cf', 0.0),
    ('resnet1202_cifar100', '2156', '28fcf78635c21d23b018d70a812eeae2ae24ad39', 'v0.0.438', 'cf', 0.0),
    ('preresnet20_cifar10', '0651', 'd3e7771e923032393bb6fa88d62625f3da64d9fe', 'v0.0.439', 'cf', 0.0),
    ('preresnet20_cifar100', '3022', '447255f8c6ad79dc42a2644438e35bc39fdeed36', 'v0.0.439', 'cf', 0.0),
    ('preresnet20_svhn', '0322', '6dcae6129ca6839c35a1ae9b3d69c4d41591811d', 'v0.0.439', 'cf', 0.0),
    ('preresnet56_cifar10', '0449', 'b4bfdaa8eaa4370899d1fb0c3c360158cf3fa3f4', 'v0.0.439', 'cf', 0.0),
    ('preresnet56_cifar100', '2505', '180fc2081f3c694b0c3db2948cb05e06f1070ee2', 'v0.0.439', 'cf', 0.0),
    ('preresnet56_svhn', '0280', '6e074c73832de7afcb8e61405b2eb62bc969d35f', 'v0.0.439', 'cf', 0.0),
    ('preresnet110_cifar10', '0386', '287a4b0cdd424fdf29d862b411f556f3d8f57f98', 'v0.0.439', 'cf', 0.0),
    ('preresnet110_cifar100', '2267', 'ab677c09518f0b7aae855153fc820811bd530c28', 'v0.0.439', 'cf', 0.0),
    ('preresnet110_svhn', '0279', '226a0b342145852f4289630f6fd82d2c90f38e01', 'v0.0.439', 'cf', 0.0),
    ('preresnet164bn_cifar10', '0364', '29a459fad0f60028b48f1908970d3947728d76b0', 'v0.0.439', 'cf', 0.0),
    ('preresnet164bn_cifar100', '2018', 'c764970119e627e5c88fe3c7cb6a7d36cd7f29d0', 'v0.0.439', 'cf', 0.0),
    ('preresnet164bn_svhn', '0258', '2307c36f351e22d9bf0240fdcf5b5651dce03e57', 'v0.0.439', 'cf', 0.0),
    ('preresnet272bn_cifar10', '0325', '5bacdc955e8d800e08d6513a6ecd21ce79da6c84', 'v0.0.439', 'cf', 0.0),
    ('preresnet272bn_cifar100', '1963', '22e0919886949484354b5a18f6c87ab5aa33b61a', 'v0.0.439', 'cf', 0.0),
    ('preresnet272bn_svhn', '0234', '3451d5fbc8dfecf2da2e624319f0e0068091f358', 'v0.0.439', 'cf', 0.0),
    ('preresnet542bn_cifar10', '0314', 'd8324d47e327c92f3557db4ba806071041a56f69', 'v0.0.439', 'cf', 0.0),
    ('preresnet542bn_cifar100', '1871', '703875c6827c83e26e05cd3e516b5a3234d01747', 'v0.0.439', 'cf', 0.0),
    ('preresnet542bn_svhn', '0236', '5ca0759231c9a045df4ef40a47d8b81e624664f8', 'v0.0.439', 'cf', 0.0),
    ('preresnet1001_cifar10', '0265', '978844c1315a0a3f6261393bcc954cecb85c199a', 'v0.0.439', 'cf', 0.0),
    ('preresnet1001_cifar100', '1841', '7481e79c54d9a32d163c740eb53310c6a5f40b01', 'v0.0.439', 'cf', 0.0),
    ('preresnet1202_cifar10', '0339', 'ab04c456454c933245d91f36942166d45393a8bc', 'v0.0.439', 'cf', 0.0),
    ('resnext20_1x64d_cifar10', '0433', 'e0ab86674852a3c78f4a600e9e8ca50a06ff0bb9', 'v0.0.440', 'cf', 0.0),
    ('resnext20_1x64d_cifar100', '2197', '413945af9f271e173bb2085de38d65e98905f304', 'v0.0.440', 'cf', 0.0),
    ('resnext20_1x64d_svhn', '0298', '105736c8c2cb1bf8a4ac4538ccd7e139501095d6', 'v0.0.440', 'cf', 0.0),
    ('resnext20_2x32d_cifar10', '0453', '7aa966dd0803c3f731d0f858125baedca245cf86', 'v0.0.440', 'cf', 0.0),
    ('resnext20_2x32d_cifar100', '2255', 'bf34e56aea7d21fca0b99c14558d6b06aab1f94f', 'v0.0.440', 'cf', 0.0),
    ('resnext20_2x32d_svhn', '0296', 'b61e1395c12285ca0c765f3ddbfd8a5c4d252536', 'v0.0.440', 'cf', 0.0),
    ('resnext20_2x64d_cifar10', '0403', '367377ed36b429753d727369cba42db281b40443', 'v0.0.440', 'cf', 0.0),
    ('resnext20_2x64d_cifar100', '2060', '6eef33bcb44c73dfdfe51036f5d647b5eba286c5', 'v0.0.440', 'cf', 0.0),
    ('resnext20_2x64d_svhn', '0283', 'dedfbac24ad3e67c55609b79da689e01ad6ba759', 'v0.0.440', 'cf', 0.0),
    ('resnext20_4x16d_cifar10', '0470', '333e834da705f54958887ce7a34335b0e71fcfad', 'v0.0.440', 'cf', 0.0),
    ('resnext20_4x16d_cifar100', '2304', 'fa8d4e06a0455f49da492377be9fe90140795629', 'v0.0.440', 'cf', 0.0),
    ('resnext20_4x16d_svhn', '0317', 'cab6d9fd851d47f11863075e83dd699cddb21571', 'v0.0.440', 'cf', 0.0),
    ('resnext20_4x32d_cifar10', '0373', 'e4aa1b0dade046bbfc872f4c84ac5fe3bcbeda11', 'v0.0.440', 'cf', 0.0),
    ('resnext20_4x32d_cifar100', '2131', 'edabd5da34edfba348b8f1712bbb0dc3ce6c5a82', 'v0.0.440', 'cf', 0.0),
    ('resnext20_4x32d_svhn', '0298', '82b75cbb31f2ea3497548a19fdf1f5fb0531527c', 'v0.0.440', 'cf', 0.0),
    ('resnext20_8x8d_cifar10', '0466', '1dbd9f5e45f120c697d128558b4d263f2ac94f0e', 'v0.0.440', 'cf', 0.0),
    ('resnext20_8x8d_cifar100', '2282', '51922108355f86cb0131826715cef9e81513e399', 'v0.0.440', 'cf', 0.0),
    ('resnext20_8x8d_svhn', '0318', '6ef55252a46d6106a160d87da107a1293cbce654', 'v0.0.440', 'cf', 0.0),
    ('resnext20_8x16d_cifar10', '0404', '5329db5f6066a73e085805ab40969af31a43e4f7', 'v0.0.440', 'cf', 0.0),
    ('resnext20_8x16d_cifar100', '2172', '3665fda790f0164078ffd6403e022a0ba8186c47', 'v0.0.440', 'cf', 0.0),
    ('resnext20_8x16d_svhn', '0301', 'd1a547e4514e6338934b26c473061b49c669c632', 'v0.0.440', 'cf', 0.0),
    ('resnext20_16x4d_cifar10', '0404', 'c671993585f1cc878941475e87c266c8a1895ca8', 'v0.0.440', 'cf', 0.0),
    ('resnext20_16x4d_cifar100', '2282', 'e800aabb6ea23a0555d2ac5a1856d7d289a46bca', 'v0.0.440', 'cf', 0.0),
    ('resnext20_16x4d_svhn', '0321', '77a670a80e976b173272614cf9416e904f1defde', 'v0.0.440', 'cf', 0.0),
    ('resnext20_16x8d_cifar10', '0394', 'cf7c675c52499a714fb3391c0240c265d6f1bb01', 'v0.0.440', 'cf', 0.0),
    ('resnext20_16x8d_cifar100', '2173', '0a33029811f76f93e79b83bf6cb19d74711c2e5b', 'v0.0.440', 'cf', 0.0),
    ('resnext20_16x8d_svhn', '0293', '4ebac2762e92f1c12b28e3012c171333a63706e1', 'v0.0.440', 'cf', 0.0),
    ('resnext20_32x2d_cifar10', '0461', 'b05d34915134060c39ea4f6b9e356b539a1e147b', 'v0.0.440', 'cf', 0.0),
    ('resnext20_32x2d_cifar100', '2322', '2def8cc21fe9057a63aee6aef2c718720fd90230', 'v0.0.440', 'cf', 0.0),
    ('resnext20_32x2d_svhn', '0327', '0c099194b551bf0d72a0028a13a94a7ca277473b', 'v0.0.440', 'cf', 0.0),
    ('resnext20_32x4d_cifar10', '0420', '6011e9e91f901ab98107e451149065524d2acc30', 'v0.0.440', 'cf', 0.0),
    ('resnext20_32x4d_cifar100', '2213', '9508c15dddd01d0064938023904c6c23ad901da5', 'v0.0.440', 'cf', 0.0),
    ('resnext20_32x4d_svhn', '0309', 'c8a843e1a0ce40fe2f42e3406e671e9a0df55d82', 'v0.0.440', 'cf', 0.0),
    ('resnext20_64x1d_cifar10', '0493', 'a13300cea5f2c626c096ac1fbf9f707a6da46f0b', 'v0.0.440', 'cf', 0.0),
    ('resnext20_64x1d_cifar100', '2353', '91695baa3caba28fa7507b3ffa0629048e01aa6e', 'v0.0.440', 'cf', 0.0),
    ('resnext20_64x1d_svhn', '0342', 'a3bad459c16926727190d1875ae90e709d50145e', 'v0.0.440', 'cf', 0.0),
    ('resnext20_64x2d_cifar10', '0438', '3846d7a7ecea5fe4da1d0895da05b675b84e23d7', 'v0.0.440', 'cf', 0.0),
    ('resnext20_64x2d_cifar100', '2235', 'e4a559ccaba13da694828aca7f83bafc9e364dcd', 'v0.0.440', 'cf', 0.0),
    ('resnext20_64x2d_svhn', '0314', 'c755e25d61534ec355c2da1a458dc5772d1f790e', 'v0.0.440', 'cf', 0.0),
    ('resnext29_16x64d_cifar10', '0241', '712e474493fd9f504010ca0a8eb10a94431bffdb', 'v0.0.440', 'cf', 0.0),
    ('resnext29_16x64d_cifar100', '1693', '2df09272ed462101da32619e652074f8c1f3ec23', 'v0.0.440', 'cf', 0.0),
    ('resnext29_16x64d_svhn', '0268', 'c929fadabc9bd8c2b2e97d4e2703ec2fba31032b', 'v0.0.440', 'cf', 0.0),
    ('resnext29_32x4d_cifar10', '0315', '5ed2e0f0945e138c3aa0c9acc0c5fd08f2d840cd', 'v0.0.440', 'cf', 0.0),
    ('resnext29_32x4d_cifar100', '1950', 'e99791392f0930372efefbe0a54304230ac4cc90', 'v0.0.440', 'cf', 0.0),
    ('resnext29_32x4d_svhn', '0280', 'de6cba99c40a882e98d2ef002cc14d799f5bf8bc', 'v0.0.440', 'cf', 0.0),
    ('resnext56_1x64d_cifar10', '0287', '5da5fe18fdf2b55977266631e2eb4b7913e7d591', 'v0.0.440', 'cf', 0.0),
    ('resnext56_1x64d_cifar100', '1825', '727009516efca454a34a3e310608b45d4c9a4020', 'v0.0.440', 'cf', 0.0),
    ('resnext56_1x64d_svhn', '0242', 'dd7ac31ee1f1a0ffcd3049fc056e8e705cae93f0', 'v0.0.440', 'cf', 0.0),
    ('resnext56_2x32d_cifar10', '0301', '54d6f2df3a903cb23978cd674495ab1e8894ab09', 'v0.0.440', 'cf', 0.0),
    ('resnext56_2x32d_cifar100', '1786', '6639c30dd1bc152736c21c9de27823d0ce3b367c', 'v0.0.440', 'cf', 0.0),
    ('resnext56_2x32d_svhn', '0246', '61524d8aff0534121257ec5b8b65647cbdafda7f', 'v0.0.440', 'cf', 0.0),
    ('resnext56_4x16d_cifar10', '0311', '766ab89fccd5b2675d5d42a9372346fd7bf45b77', 'v0.0.440', 'cf', 0.0),
    ('resnext56_4x16d_cifar100', '1809', '61b41c3b953a4a7198dec6a379f789030a998e42', 'v0.0.440', 'cf', 0.0),
    ('resnext56_4x16d_svhn', '0244', 'b7ab24694a0c1f635fbb2b2e4130272b5e75b6bc', 'v0.0.440', 'cf', 0.0),
    ('resnext56_8x8d_cifar10', '0307', '685eab396974992f71402533be96229cdc3eb751', 'v0.0.440', 'cf', 0.0),
    ('resnext56_8x8d_cifar100', '1806', 'f3f80382faa7baadaef4e09fedb924b4d5deac78', 'v0.0.440', 'cf', 0.0),
    ('resnext56_8x8d_svhn', '0247', '85692d770f3bab690dc9aa57b4e3d9aa728121e9', 'v0.0.440', 'cf', 0.0),
    ('resnext56_16x4d_cifar10', '0312', '930e5d5baf62d2fe4e48afe7dbd928079fd5531a', 'v0.0.440', 'cf', 0.0),
    ('resnext56_16x4d_cifar100', '1824', '667ba1835c3db07e54ad4dfbc6ea99a0b12afd78', 'v0.0.440', 'cf', 0.0),
    ('resnext56_16x4d_svhn', '0256', '86f327a9652e79a4a38c0d6ebc9fda8f0a6c3ea4', 'v0.0.440', 'cf', 0.0),
    ('resnext56_32x2d_cifar10', '0314', '9e387e2e6c769802fbf7a911b67d2c490e14db85', 'v0.0.440', 'cf', 0.0),
    ('resnext56_32x2d_cifar100', '1860', '7a236896b7f00913f8a0846d39382d87bc56214c', 'v0.0.440', 'cf', 0.0),
    ('resnext56_32x2d_svhn', '0253', 'b93a0535890a340774a190fab2a521696b134600', 'v0.0.440', 'cf', 0.0),
    ('resnext56_64x1d_cifar10', '0341', 'bc7469474a3cf31622186aa86c0c837b9c05563a', 'v0.0.440', 'cf', 0.0),
    ('resnext56_64x1d_cifar100', '1816', '06c6c7a0bb97cd67360e624dd9ca3193969c3e06', 'v0.0.440', 'cf', 0.0),
    ('resnext56_64x1d_svhn', '0255', '9e9e3cc2bf26b8c691b5b2b12fb3908dd999f870', 'v0.0.440', 'cf', 0.0),
    ('resnext272_1x64d_cifar10', '0255', '6efe448a89da1340dca7158d12a0355d1b2d2d75', 'v0.0.440', 'cf', 0.0),
    ('resnext272_1x64d_cifar100', '1911', 'e9275c944ff841c29316a2728068a6162af39488', 'v0.0.440', 'cf', 0.0),
    ('resnext272_1x64d_svhn', '0234', '4d348e9ec9d261318d1264c61f4817de612797e4', 'v0.0.440', 'cf', 0.0),
    ('resnext272_2x32d_cifar10', '0274', '4e35f99476d34225bd07ed2f4274ed021fb635f3', 'v0.0.440', 'cf', 0.0),
    ('resnext272_2x32d_cifar100', '1834', '274ef60797974e3d7290644861facefa983bc7f2', 'v0.0.440', 'cf', 0.0),
    ('resnext272_2x32d_svhn', '0244', 'f792396540a630a0d51932f9c7557e5d96ddb66c', 'v0.0.440', 'cf', 0.0),
    ('seresnet20_cifar10', '0601', '2f392e4a48cffe1ff96b92ca28fd0f020e9d89aa', 'v0.0.442', 'cf', 0.0),
    ('seresnet20_cifar100', '2854', '598b585838afb8907e76c6e9af2b92417f5eeb08', 'v0.0.442', 'cf', 0.0),
    ('seresnet20_svhn', '0323', 'ef43ce80cc226dff6d7c0fd120daaa89fe353392', 'v0.0.442', 'cf', 0.0),
    ('seresnet56_cifar10', '0413', '0224e930258e0567cf18bd1b0f5ae8ffd85d6231', 'v0.0.442', 'cf', 0.0),
    ('seresnet56_cifar100', '2294', '9c86ec999dac74831ab3918682c1753fde447187', 'v0.0.442', 'cf', 0.0),
    ('seresnet56_svhn', '0264', 'a8fcc570f6ab95d188148f0070f714c052bcf0f3', 'v0.0.442', 'cf', 0.0),
    ('seresnet110_cifar10', '0363', '4c28f93f8fe23a216aba5cb80af8412023b42cdb', 'v0.0.442', 'cf', 0.0),
    ('seresnet110_cifar100', '2086', '6435b022d058e62f95bbd2bb6447cd76f0a14316', 'v0.0.442', 'cf', 0.0),
    ('seresnet110_svhn', '0235', '57751ac70c94c9bbe95a1229af30b5471db498b1', 'v0.0.442', 'cf', 0.0),
    ('seresnet164bn_cifar10', '0339', '64d051543b02cb26fb6a22220ad35bb5b80243e3', 'v0.0.442', 'cf', 0.0),
    ('seresnet164bn_cifar100', '1995', '121a777aa64b7249a9483baa1e8a677a7c9587df', 'v0.0.442', 'cf', 0.0),
    ('seresnet164bn_svhn', '0245', 'a19e2e88575459f35303a058e486a944e34f8379', 'v0.0.442', 'cf', 0.0),
    ('seresnet272bn_cifar10', '0339', 'baa561b6c4449558a11900ae24780d6fcdd9efdf', 'v0.0.442', 'cf', 0.0),
    ('seresnet272bn_cifar100', '1907', 'a29e50de59aac03cff1d657ce0653a02246c39dc', 'v0.0.442', 'cf', 0.0),
    ('seresnet272bn_svhn', '0238', '918ee0dea7a956bca36d23459e822488e3a0659e', 'v0.0.442', 'cf', 0.0),
    ('seresnet542bn_cifar10', '0347', 'e95ebdb9b79f4955731147c078e1607dd174ffe9', 'v0.0.442', 'cf', 0.0),
    ('seresnet542bn_cifar100', '1887', 'ddc4d5c89d56a0c560e5174194db071fcb960d81', 'v0.0.442', 'cf', 0.0),
    ('seresnet542bn_svhn', '0226', '5ec784aabe3030f519ca22821b7a58a30e0bf179', 'v0.0.442', 'cf', 0.0),
    ('sepreresnet20_cifar10', '0618', '22217b323af922b720bc044bce9556b0dde18d97', 'v0.0.443', 'cf', 0.0),
    ('sepreresnet20_cifar100', '2831', 'e8dab8b87dbe512dfabd7cdbaff9b08be81fb36b', 'v0.0.443', 'cf', 0.0),
    ('sepreresnet20_svhn', '0324', 'e7dbcc9678dfa8ce0b2699de601699d29a5cb868', 'v0.0.443', 'cf', 0.0),
    ('sepreresnet56_cifar10', '0451', '32637db56c6fed2a3d66778ee3335527f2d8e25d', 'v0.0.443', 'cf', 0.0),
    ('sepreresnet56_cifar100', '2305', 'aea4d90bc7fd0eb8f433e376d1aba8e3c0d1ac55', 'v0.0.443', 'cf', 0.0),
    ('sepreresnet56_svhn', '0271', 'ea024196ca9bd0ff331e8d8d3da376aecf9ea0c1', 'v0.0.443', 'cf', 0.0),
    ('sepreresnet110_cifar10', '0454', 'e317c56922fbf1cec478e46e49d6edd3c4ae3b03', 'v0.0.443', 'cf', 0.0),
    ('sepreresnet110_cifar100', '2261', '19a8d4a1563f8fb61c63a5c577f40f3363efec00', 'v0.0.443', 'cf', 0.0),
    ('sepreresnet110_svhn', '0259', '6291c548277580f90ed0e22845f06eb7b022f8f9', 'v0.0.443', 'cf', 0.0),
    ('sepreresnet164bn_cifar10', '0373', '253c0430d6e8d2ba9c4c5526beed3b2e90573fe4', 'v0.0.443', 'cf', 0.0),
    ('sepreresnet164bn_cifar100', '2005', '9c3ed25062e52a23f73600c1a0f99064f89b4a47', 'v0.0.443', 'cf', 0.0),
    ('sepreresnet164bn_svhn', '0256', 'c89523226a8a010459ebec9c48d940773946e7bf', 'v0.0.443', 'cf', 0.0),
    ('sepreresnet272bn_cifar10', '0339', '1ca0bed3b3ae20d55322fa2f75057edb744fb63d', 'v0.0.443', 'cf', 0.0),
    ('sepreresnet272bn_cifar100', '1913', 'eb75217f625dbc97af737e5878a9eab28fdf3b03', 'v0.0.443', 'cf', 0.0),
    ('sepreresnet272bn_svhn', '0249', '0a778e9d68f6921463563ef84054969221809aef', 'v0.0.443', 'cf', 0.0),
    ('sepreresnet542bn_cifar10', '0309', '7764e8bddba21c75b8f8d4775093721d859f850c', 'v0.0.443', 'cf', 0.0),
    ('sepreresnet542bn_cifar100', '1945', '969d2bf0a8d213757486e18c180ba14058e08eac', 'v0.0.443', 'cf', 0.0),
    ('sepreresnet542bn_svhn', '0247', '8e2427367762cf20b67b407e2a1ec8479b0ad41c', 'v0.0.443', 'cf', 0.0),
    ('pyramidnet110_a48_cifar10', '0372', '3b6ab16073fb0ff438d4376d320be9b119aee362', 'v0.0.444', 'cf', 0.0),
    ('pyramidnet110_a48_cifar100', '2095', '3490690ae62adc4b91dc29ba06f9dc2abf272fce', 'v0.0.444', 'cf', 0.0),
    ('pyramidnet110_a48_svhn', '0247', '1582739049630e1665b577781ccca1e65f961749', 'v0.0.444', 'cf', 0.0),
    ('pyramidnet110_a84_cifar10', '0298', 'bf303f3414123bdf79cb23d3316dd171df74f5d4', 'v0.0.444', 'cf', 0.0),
    ('pyramidnet110_a84_cifar100', '1887', '85789d68d11ad663a53ed921ce6fb28a98248874', 'v0.0.444', 'cf', 0.0),
    ('pyramidnet110_a84_svhn', '0243', 'aacb5f882c7810181c0d4de061c2a76dfbf4925b', 'v0.0.444', 'cf', 0.0),
    ('pyramidnet110_a270_cifar10', '0251', '983d99830e7bb23ca0123ec47dfa05143eb8a37e', 'v0.0.444', 'cf', 0.0),
    ('pyramidnet110_a270_cifar100', '1710', 'cc58021f2406c3593a51f62d03fea714d0649036', 'v0.0.444', 'cf', 0.0),
    ('pyramidnet110_a270_svhn', '0238', 'b8742320795657a0b51d35226c2e14fc76acac11', 'v0.0.444', 'cf', 0.0),
    ('pyramidnet164_a270_bn_cifar10', '0242', 'aa879193cd4730fd06430b494c11497121fad2df', 'v0.0.444', 'cf', 0.0),
    ('pyramidnet164_a270_bn_cifar100', '1670', '25ddf056b681987c1db76b60a08a1e1a7830a51e', 'v0.0.444', 'cf', 0.0),
    ('pyramidnet164_a270_bn_svhn', '0234', '94bb4029e52688f7616d5fd680acacf7c6e3cd4e', 'v0.0.444', 'cf', 0.0),
    ('pyramidnet200_a240_bn_cifar10', '0244', 'c269bf7d485a13a9beed9c0aade75ff959584ef9', 'v0.0.444', 'cf', 0.0),
    ('pyramidnet200_a240_bn_cifar100', '1609', 'd2b1682287b6047477c3efd322f305957bb393ef', 'v0.0.444', 'cf', 0.0),
    ('pyramidnet200_a240_bn_svhn', '0232', '77f2380c1fd77abb80b830e0d44f2986fde28ec9', 'v0.0.444', 'cf', 0.0),
    ('pyramidnet236_a220_bn_cifar10', '0247', '26aac5d0938a96902484f0a51f7f3440551c9c96', 'v0.0.444', 'cf', 0.0),
    ('pyramidnet236_a220_bn_cifar100', '1634', '37d5b197d45c3985ad3a9ba346f148e63cd271fb', 'v0.0.444', 'cf', 0.0),
    ('pyramidnet236_a220_bn_svhn', '0235', '6a9a8b0a5fbcce177c8b4449ad138b6f3a94f2bb', 'v0.0.444', 'cf', 0.0),
    ('pyramidnet272_a200_bn_cifar10', '0239', 'b57f64f1964798fac3d62fd796c87df8132cf18c', 'v0.0.444', 'cf', 0.0),
    ('pyramidnet272_a200_bn_cifar100', '1619', '5c233384141f7700da643c53f4245d2f0d00ded7', 'v0.0.444', 'cf', 0.0),
    ('pyramidnet272_a200_bn_svhn', '0240', '0a389e2f1811af7cacc2a27b6df748a7c46d951a', 'v0.0.444', 'cf', 0.0),
    ('densenet40_k12_cifar10', '0561', 'e6e20ebfcc60330050d4c1eb94d03d8fadb738df', 'v0.0.445', 'cf', 0.0),
    ('densenet40_k12_cifar100', '2490', 'ef38ff655136f7921e785836c659be7f1d11424d', 'v0.0.445', 'cf', 0.0),
    ('densenet40_k12_svhn', '0305', '7d5860ae4c8f912a4374e6214720d13ad52f3ffb', 'v0.0.445', 'cf', 0.0),
    ('densenet40_k12_bc_cifar10', '0643', '58950791713ee0ec19f6e1bc6e6e3731fc4a9484', 'v0.0.445', 'cf', 0.0),
    ('densenet40_k12_bc_cifar100', '2841', 'c7fbb0f4e74cafbd0e329597e63fbc81682c8e90', 'v0.0.445', 'cf', 0.0),
    ('densenet40_k12_bc_svhn', '0320', '77fd3ddf577ba336f7eac64f0ac6afaabbb25fd1', 'v0.0.445', 'cf', 0.0),
    ('densenet40_k24_bc_cifar10', '0452', '61a7fe9c0654161991da1e4eb1e0286d451d8cec', 'v0.0.445', 'cf', 0.0),
    ('densenet40_k24_bc_cifar100', '2267', 'b3878e8252d7ae1c53b6d2b5c6f77a857c281e9b', 'v0.0.445', 'cf', 0.0),
    ('densenet40_k24_bc_svhn', '0290', 'b8a231f7cd23b122bb8d9afe362c6de2663c1241', 'v0.0.445', 'cf', 0.0),
    ('densenet40_k36_bc_cifar10', '0404', 'ce27624f5701f020d2feff0e88e69da07b0ef958', 'v0.0.445', 'cf', 0.0),
    ('densenet40_k36_bc_cifar100', '2050', '045ae83a5ee3d1a85864cadadeb537242138c2d8', 'v0.0.445', 'cf', 0.0),
    ('densenet40_k36_bc_svhn', '0260', 'a176dcf180f086d88bbf4ff028b084bf02394a35', 'v0.0.445', 'cf', 0.0),
    ('densenet100_k12_cifar10', '0366', 'fc483c0bdd58e5013a3910f939334d5f40c65438', 'v0.0.445', 'cf', 0.0),
    ('densenet100_k12_cifar100', '1965', '4f0083d6698d42165c8b326c1e4beda6d9679796', 'v0.0.445', 'cf', 0.0),
    ('densenet100_k12_svhn', '0260', 'e810c38067bf34dc679caaeb4021623f2277b6b8', 'v0.0.445', 'cf', 0.0),
    ('densenet100_k24_cifar10', '0313', '7f9ee9b3787c2540c4448f424c504f0509000234', 'v0.0.445', 'cf', 0.0),
    ('densenet100_k24_cifar100', '1808', 'b0842c59c00f14df58d0f8bbac8348837e30e751', 'v0.0.445', 'cf', 0.0),
    ('densenet100_k12_bc_cifar10', '0416', '66beb8fc89f7e40d2b529e0f3270549324b5b784', 'v0.0.445', 'cf', 0.0),
    ('densenet100_k12_bc_cifar100', '2119', 'c1b857d51eb582eee8dbd7250d05871e40a7f4c4', 'v0.0.445', 'cf', 0.0),
    ('densenet190_k40_bc_cifar10', '0252', '9cc5cfcbef9425227370ac8c6404cfc1e3edbf55', 'v0.0.445', 'cf', 0.0),
    ('densenet250_k24_bc_cifar10', '0267', '3217a1b3c61afc9d08bc4b43bff4aac103da0012', 'v0.0.445', 'cf', 0.0),
    ('densenet250_k24_bc_cifar100', '1739', '02d967b564c48b25117aac6cd7b095fd5d30d4d5', 'v0.0.445', 'cf', 0.0),
    ('resnet10_cub', '2758', '1a6846b3854d1942997d7082e94b330ddce3db19', 'v0.0.446', 'cub', 0.0),
    ('resnet12_cub', '2668', '03c8073655ae51f21ceed7d7f86f9ed6169fc310', 'v0.0.446', 'cub', 0.0),
    ('resnet14_cub', '2435', '24b0bfebaa0d1b4442fa63a659d22de8ff594118', 'v0.0.446', 'cub', 0.0),
    ('resnet16_cub', '2328', '81cc8192c880c687175d636a0339e16463c61627', 'v0.0.446', 'cub', 0.0),
    ('resnet18_cub', '2335', '198bdc26bbfaad777ea6d494c41b9d66a493aac7', 'v0.0.446', 'cub', 0.0),
    ('resnet26_cub', '2264', '545967849063af9b5ec55a5cf339f5897f394e85', 'v0.0.446', 'cub', 0.0),
    ('seresnet10_cub', '2749', '484fc1661dda247db32dd6a54b88dc156da5156c', 'v0.0.446', 'cub', 0.0),
    ('seresnet12_cub', '2611', '0e5b4e23f30add924f8cad41704cb335a36b2049', 'v0.0.446', 'cub', 0.0),
    ('seresnet14_cub', '2375', '56c268728f7343aa1410cb2f046860c34428b123', 'v0.0.446', 'cub', 0.0),
    ('seresnet16_cub', '2321', 'ed3ead791be4af44aa1202f0dbf4b26fdb770963', 'v0.0.446', 'cub', 0.0),
    ('seresnet18_cub', '2309', 'f699f05f2a2ce41dae01d5d6c180ec2569356f0a', 'v0.0.446', 'cub', 0.0),
    ('seresnet26_cub', '2258', 'c02ba47493bc9185a7fb06584e23b5a740082e77', 'v0.0.446', 'cub', 0.0),
    ('mobilenet_w1_cub', '2346', 'b8f24c14b9ed9629efb161510547e30c4a37edc2', 'v0.0.446', 'cub', 0.0),
    ('proxylessnas_mobile_cub', '2202', '73ceed5a6a3f870b306da0c48318d969e53d6340', 'v0.0.446', 'cub', 0.0),
    ('pspnet_resnetd101b_voc', '7599', 'fbe47bfce77b8c9cab3c9c5913f6a42c04cce946', 'v0.0.448', 'voc', 0.0),
    ('pspnet_resnetd50b_ade20k', '2712', 'f4fadf0b3f5a39e1ab070736d792bd9259c0d371', 'v0.0.450', 'voc', 0.0),
    ('pspnet_resnetd101b_ade20k', '3259', 'ac8569f44bd646ee8875d2b3eae0ab54c72c4904', 'v0.0.450', 'voc', 0.0),
    ('pspnet_resnetd101b_coco', '5438', 'b64ff2dcde6d3f989c45cec2a021d3769f4cb9eb', 'v0.0.451', 'voc', 0.0),
    ('pspnet_resnetd101b_cityscapes', '5760', '6dc20af68e9de31b663469b170e75cb016bd3a1f', 'v0.0.449', 'cs', 0.0),
    ('deeplabv3_resnetd101b_voc', '7560', 'e261b6fd9c4878c41bfa088777ea53fcddb4fa51', 'v0.0.448', 'voc', 0.0),
    ('deeplabv3_resnetd152b_voc', '7791', '72038caba5f552c77d08ad768bda004643f1c53e', 'v0.0.448', 'voc', 0.0),
    ('deeplabv3_resnetd50b_ade20k', '3172', '2ba069a73d81d6b2ceaf7f2c57f2fe3dd673b78b', 'v0.0.450', 'voc', 0.0),
    ('deeplabv3_resnetd101b_ade20k', '3488', '08c90933a65061a56e3b22e9c143340a98455075', 'v0.0.450', 'voc', 0.0),
    ('deeplabv3_resnetd101b_coco', '5865', '39525a1333ebf12ca32578f32831b3e5b22a887a', 'v0.0.451', 'voc', 0.0),
    ('deeplabv3_resnetd152b_coco', '6067', 'f4dabc62dc8209e7a9adf0dceef97837b06b21c9', 'v0.0.451', 'voc', 0.0),
    ('fcn8sd_resnetd101b_voc', '8039', 'e140349ce60ad3943b535efb081b3e9c2a58f6e9', 'v0.0.448', 'voc', 0.0),
    ('fcn8sd_resnetd50b_ade20k', '3310', 'd440f859bad1c84790aa1c3e1c0addc21b171d4a', 'v0.0.450', 'voc', 0.0),
    ('fcn8sd_resnetd101b_ade20k', '3550', '970d968a1fb44670993b065c1603a6a7c0bd57a1', 'v0.0.450', 'voc', 0.0),
    ('fcn8sd_resnetd101b_coco', '5968', '69c001b3875c5399dfc1281eb5a051bafef40e4b', 'v0.0.451', 'voc', 0.0),
    ('icnet_resnetd50b_cityscapes', '6060', '1e53e1d1724e61cc740cfbc818ca6e14015185ef', 'v0.0.457', 'cs', 0.0),
    ('bisenet_resnet18_celebamaskhq', '0000', 'e8799341e74332932f5d162e3c1c780596caa219', 'v0.0.462', 'cs', 0.0),
    ('alphapose_fastseresnet101b_coco', '7415', 'd1f0464a0f2c520d8690d49d09fe1426b0ab3eab', 'v0.0.454', 'cocohpe', 0.0),
    ('simplepose_resnet18_coco', '6631', '4d907c70a6f3ccaba321c05406ce038351e0c67f', 'v0.0.455', 'cocohpe', 0.0),
    ('simplepose_resnet50b_coco', '7102', '74506b66735333e3deab5908d309d3ec04c94861', 'v0.0.455', 'cocohpe', 0.0),
    ('simplepose_resnet101b_coco', '7244', '6f9e08d6afa08e83176e8e04f7566e255265e080', 'v0.0.455', 'cocohpe', 0.0),
    ('simplepose_resnet152b_coco', '7253', 'c018fb87bb8e5f5d8d6daa6a922869b2f36481cf', 'v0.0.455', 'cocohpe', 0.0),
    ('simplepose_resneta50b_coco', '7170', 'c9ddc1c90ddac88b1f64eb962e1bda87887668a5', 'v0.0.455', 'cocohpe', 0.0),
    ('simplepose_resneta101b_coco', '7297', '6db62b714be632359020c972bedb459e5210820f', 'v0.0.455', 'cocohpe', 0.0),
    ('simplepose_resneta152b_coco', '7344', 'f65954b9df20bf9fa64a9791563729fa51983cf5', 'v0.0.455', 'cocohpe', 0.0),
    ('simplepose_mobile_resnet18_coco', '6625', '8f3e5cc4c6af306c23f0882887d7b36ee0b1079a', 'v0.0.456', 'cocohpe', 0.0),  # noqa
    ('simplepose_mobile_resnet50b_coco', '7110', 'e8f61fdaf7aacbe58d006129943988ae95c9aef3', 'v0.0.456', 'cocohpe', 0.0),  # noqa
    ('simplepose_mobile_mobilenet_w1_coco', '6410', '27c918b95148b87944eec36ac422bf18792513ae', 'v0.0.456', 'cocohpe', 0.0),  # noqa
    ('simplepose_mobile_mobilenetv2b_w1_coco', '6374', '4bcc3462fb2af46ed6daed78d15920a274e58051', 'v0.0.456', 'cocohpe', 0.0),  # noqa
    ('simplepose_mobile_mobilenetv3_small_w1_coco', '5434', '1cfee871467e99e7af23e5135bb9a4765f010a05', 'v0.0.456', 'cocohpe', 0.0),  # noqa
    ('simplepose_mobile_mobilenetv3_large_w1_coco', '6367', '8c8583fbe6d60355c232a10b5de8a455a38ba073', 'v0.0.456', 'cocohpe', 0.0),  # noqa
    ('lwopenpose2d_mobilenet_cmupan_coco', '3999', '626b66cb1d36d0721b59d5acaa8d08d7690ea830', 'v0.0.458', 'cocohpe', 0.0),  # noqa
    ('lwopenpose3d_mobilenet_cmupan_coco', '3999', 'df9b1c5f667deb93a87f69479ce92093e7c9f3b6', 'v0.0.458', 'cocohpe', 0.0),  # noqa
    ('ibppose_coco', '6487', '79500f3d5dd990fd63544e3e3ca65f0382b06e44', 'v0.0.459', 'cocohpe', 0.0),
]}

imgclsmob_repo_url = 'https://github.com/osmr/imgclsmob'


def get_model_name_suffix_data(model_name):
    if model_name not in _model_sha1:
        raise ValueError("Pretrained model for {name} is not available.".format(name=model_name))
    error, sha1_hash, repo_release_tag, ds, scale = _model_sha1[model_name]
    return error, sha1_hash, repo_release_tag


def get_model_file(model_name,
                   local_model_store_dir_path=os.path.join("~", ".tensorflow", "models")):
    """
    Return location for the pretrained on local file system. This function will download from online model zoo when
    model cannot be found or has mismatch. The root directory will be created if it doesn't exist.

    Parameters
    ----------
    model_name : str
        Name of the model.
    local_model_store_dir_path : str, default $TENSORFLOW_HOME/models
        Location for keeping the model parameters.

    Returns
    -------
    file_path
        Path to the requested pretrained model file.
    """
    error, sha1_hash, repo_release_tag = get_model_name_suffix_data(model_name)
    short_sha1 = sha1_hash[:8]
    file_name = "{name}-{error}-{short_sha1}.tf2.h5".format(
        name=model_name,
        error=error,
        short_sha1=short_sha1)
    local_model_store_dir_path = os.path.expanduser(local_model_store_dir_path)
    file_path = os.path.join(local_model_store_dir_path, file_name)
    if os.path.exists(file_path):
        if _check_sha1(file_path, sha1_hash):
            return file_path
        else:
            logging.warning("Mismatch in the content of model file detected. Downloading again.")
    else:
        logging.info("Model file not found. Downloading to {}.".format(file_path))

    if not os.path.exists(local_model_store_dir_path):
        os.makedirs(local_model_store_dir_path)

    zip_file_path = file_path + ".zip"
    _download(
        url="{repo_url}/releases/download/{repo_release_tag}/{file_name}.zip".format(
            repo_url=imgclsmob_repo_url,
            repo_release_tag=repo_release_tag,
            file_name=file_name),
        path=zip_file_path,
        overwrite=True)
    with zipfile.ZipFile(zip_file_path) as zf:
        zf.extractall(local_model_store_dir_path)
    os.remove(zip_file_path)

    if _check_sha1(file_path, sha1_hash):
        return file_path
    else:
        raise ValueError("Downloaded file has different hash. Please try again.")


def _download(url, path=None, overwrite=False, sha1_hash=None, retries=5, verify_ssl=True):
    """
    Download an given URL

    Parameters
    ----------
    url : str
        URL to download
    path : str, optional
        Destination path to store downloaded file. By default stores to the
        current directory with same name as in url.
    overwrite : bool, optional
        Whether to overwrite destination file if already exists.
    sha1_hash : str, optional
        Expected sha1 hash in hexadecimal digits. Will ignore existing file when hash is specified
        but doesn't match.
    retries : integer, default 5
        The number of times to attempt the download in case of failure or non 200 return codes
    verify_ssl : bool, default True
        Verify SSL certificates.

    Returns
    -------
    str
        The file path of the downloaded file.
    """
    import warnings
    try:
        import requests
    except ImportError:
        class requests_failed_to_import(object):
            pass
        requests = requests_failed_to_import

    if path is None:
        fname = url.split("/")[-1]
        # Empty filenames are invalid
        assert fname, "Can't construct file-name from this URL. Please set the `path` option manually."
    else:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            fname = os.path.join(path, url.split("/")[-1])
        else:
            fname = path
    assert retries >= 0, "Number of retries should be at least 0"

    if not verify_ssl:
        warnings.warn(
            "Unverified HTTPS request is being made (verify_ssl=False). "
            "Adding certificate verification is strongly advised.")

    if overwrite or not os.path.exists(fname) or (sha1_hash and not _check_sha1(fname, sha1_hash)):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        while retries + 1 > 0:
            # Disable pyling too broad Exception
            # pylint: disable=W0703
            try:
                print("Downloading {} from {}...".format(fname, url))
                r = requests.get(url, stream=True, verify=verify_ssl)
                if r.status_code != 200:
                    raise RuntimeError("Failed downloading url {}".format(url))
                with open(fname, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
                if sha1_hash and not _check_sha1(fname, sha1_hash):
                    raise UserWarning("File {} is downloaded but the content hash does not match."
                                      " The repo may be outdated or download may be incomplete. "
                                      "If the `repo_url` is overridden, consider switching to "
                                      "the default repo.".format(fname))
                break
            except Exception as e:
                retries -= 1
                if retries <= 0:
                    raise e
                else:
                    print("download failed, retrying, {} attempt{} left"
                          .format(retries, "s" if retries > 1 else ""))

    return fname


def _check_sha1(filename, sha1_hash):
    """
    Check whether the sha1 hash of the file content matches the expected hash.

    Parameters
    ----------
    filename : str
        Path to the file.
    sha1_hash : str
        Expected sha1 hash in hexadecimal digits.

    Returns
    -------
    bool
        Whether the file content matches the expected hash.
    """
    sha1 = hashlib.sha1()
    with open(filename, "rb") as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)

    return sha1.hexdigest() == sha1_hash