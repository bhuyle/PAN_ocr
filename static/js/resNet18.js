$(document).ready(function () {
    // Init
    $('.image-section-ResNet50').hide();
    $('.loaderResNet50').hide();
    $('#resultResNet18').hide();

    // Upload Preview
    function readURLResNet50(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreviewResNet50').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreviewResNet50').hide();
                $('#imagePreviewResNet50').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUploadResNet50").change(function () {
        $('.image-section-ResNet50').show();
        $('#btn-predict-ResNet50').show();
        $('#resultResNet18').text('');
        $('#resultResNet18').hide();
        $('#text18').text('')
        readURLResNet50(this);
    });

    // Predict
    $('#btn-predict-ResNet50').click(function () {
        var form_data = new FormData($('#upload-file-ResNet50')[0]);

        // Show loading animation
        $(this).hide();
        $('.loaderResNet50').show();
        console.log("RESNET 18")
        // Make prediction by calling api /predictResNet50
        $.ajax({
            type: 'POST',
            url: './retrained/predictResNet18',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loaderResNet50').hide();
                $('#resultResNet18').fadeIn(600);
                $("#resultResNet18").attr("src","data:image/png;base64,"+ data.image);
                $('#text18').html("Number of bboxes "+data.box+'<br>Time '+data.time)

            },
        });
    });

});