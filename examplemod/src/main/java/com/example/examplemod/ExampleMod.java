package com.example.examplemod;

import com.mojang.logging.LogUtils;
import net.minecraft.core.registries.Registries;
import net.minecraft.network.chat.Component;
import net.minecraft.world.item.CreativeModeTab;
import net.minecraft.world.item.CreativeModeTabs;
import net.neoforged.bus.api.IEventBus;
import net.neoforged.bus.api.SubscribeEvent;
import net.neoforged.fml.ModContainer;
import net.neoforged.fml.common.Mod;
import net.neoforged.fml.config.ModConfig;
import net.neoforged.fml.event.lifecycle.FMLCommonSetupEvent;
import net.neoforged.neoforge.common.NeoForge;
import net.neoforged.neoforge.event.BuildCreativeModeTabContentsEvent;
import net.neoforged.neoforge.event.server.ServerStartingEvent;
import net.neoforged.neoforge.registries.DeferredRegister;
import org.slf4j.Logger;

@Mod(ExampleMod.MODID)
public class ExampleMod {
    public static final String MODID = "examplemod";
    public static final Logger LOGGER = LogUtils.getLogger();

    // 核心注册器
    public static final DeferredRegister.Blocks BLOCKS = DeferredRegister.createBlocks(MODID);
    public static final DeferredRegister.Items ITEMS = DeferredRegister.createItems(MODID);
    public static final DeferredRegister<CreativeModeTab> CREATIVE_MODE_TABS = DeferredRegister.create(Registries.CREATIVE_MODE_TAB, MODID);

    // 注册创造模式标签页
    public static final net.neoforged.neoforge.registries.DeferredHolder<CreativeModeTab, CreativeModeTab> EXAMPLE_TAB = CREATIVE_MODE_TABS.register("example_tab", () -> CreativeModeTab.builder()
            .title(Component.translatable("itemGroup.examplemod"))
            .withTabsBefore(CreativeModeTabs.COMBAT)
            // 注意：这里需要给 Tab 指定一个图标，这里暂时先随便指定一个，
            // 实际上你可以在 GeneratedBlocks 里生成完之后，在这里引用其中一个作为图标
            .icon(() -> net.minecraft.world.item.Items.STONE.getDefaultInstance())
            .displayItems((parameters, output) -> {
                // 调用自动生成的代码，把所有物品加进去
                GeneratedBlocks.registerItemsToTab(output);
            }).build());

    // 在 ExampleMod.java 中

    public ExampleMod(IEventBus modEventBus, ModContainer modContainer) {
        modEventBus.addListener(this::commonSetup);

        // === 核心修改在这里 ===
        // 必须在注册之前或同时，强制加载生成的类
        // 这样 GeneratedBlocks 里的 static 代码才会运行，方块才会加入 BLOCKS 队列
        GeneratedBlocks.init();
        // ====================

        // 注册核心总线
        BLOCKS.register(modEventBus);
        ITEMS.register(modEventBus);
        CREATIVE_MODE_TABS.register(modEventBus);

        NeoForge.EVENT_BUS.register(this);
        modEventBus.addListener(this::addCreative);
        modContainer.registerConfig(ModConfig.Type.COMMON, Config.SPEC);
    }

    private void commonSetup(FMLCommonSetupEvent event) {
        LOGGER.info("COMMON SETUP");
    }

    private void addCreative(BuildCreativeModeTabContentsEvent event) {
        // 如果你也想把物品加到原版的“建筑方块”栏里
        if (event.getTabKey() == CreativeModeTabs.BUILDING_BLOCKS) {
            GeneratedBlocks.registerItemsToTab(event);
        }
    }

    @SubscribeEvent
    public void onServerStarting(ServerStartingEvent event) {
        LOGGER.info("Server starting...");
    }
}